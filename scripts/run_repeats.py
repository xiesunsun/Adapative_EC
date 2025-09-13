#!/usr/bin/env python3
"""
Run the GA multiple times on a single dataset config and aggregate results.

Metrics reported (across repeats):
- Avg_fitness: mean of best_solution.json.best_fitness
- Avg_accuracy: mean of best_solution.json.cv_mean
- Selected_num: mean of best_solution.json.num_selected
- Avg_max_fitness_step: for each run, find earliest generation where 'max' in evolution_log.csv reaches its run-maximum; average across runs
- Avg_diversity: for each run, average 'diversity' across generations (ignore NaN); then average across runs
- Best Fitness: max of best_solution.json.best_fitness across runs
- Best Accuracy: max of best_solution.json.cv_mean across runs

Usage example:
  python scripts/run_repeats.py \
    --config-dir config \
    --out-root results/repeats_demo \
    --repeats 3 \
    --seed 42 \
    --with-aos --aos-strict

Notes:
- Uses --use-config to respect algo_config.json & operator_pools.json in the config dir.
- Outputs each run to <out-root>/run_<i> and writes a summary CSV to <out-root>/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import List, Dict, Tuple


def _safe_float(x) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return float('nan')
    except Exception:
        return float('nan')


def run_one(config_dir: str, out_dir: str, seed: int, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "feature_selection_ga.py", "--use-config",
        "--config-dir", config_dir,
        "--output", out_dir,
        "--seed", str(seed),
    ]
    if args.aos_enable:
        cmd.append("--aos-enable")
    if args.aos_strict:
        cmd.append("--aos-strict")
    for ea in (args.extra_arg or []):
        cmd.append(ea)
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)


def parse_best(best_path: str) -> Tuple[float, float, int]:
    with open(best_path, "r") as f:
        data = json.load(f)
    return (
        float(data.get("best_fitness", float("nan"))),
        float(data.get("cv_mean", float("nan"))),
        int(data.get("num_selected", 0)),
    )


def parse_evolution(evo_csv: str) -> Tuple[float, float]:
    """Return (earliest_max_gen, avg_diversity)."""
    if not os.path.exists(evo_csv):
        return float("nan"), float("nan")
    gens: List[int] = []
    max_vals: List[float] = []
    div_vals: List[float] = []
    with open(evo_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                g = int(row.get("gen", "0"))
            except Exception:
                continue
            gens.append(g)
            max_vals.append(_safe_float(row.get("max", "nan")))
            div_vals.append(_safe_float(row.get("diversity", "nan")))
    if not gens or not max_vals:
        return float("nan"), float("nan")
    # earliest generation attaining run-maximum
    run_max = max([v for v in max_vals if v == v], default=float("nan"))
    earliest = float("nan")
    if run_max == run_max:
        for g, v in zip(gens, max_vals):
            if v == v and abs(v - run_max) <= 1e-12:
                earliest = float(g)
                break
    # avg diversity
    div_clean = [v for v in div_vals if v == v]
    avg_div = sum(div_clean) / len(div_clean) if div_clean else float("nan")
    return earliest, avg_div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=str, required=True, help="Directory containing task_info.json, algo_config.json, operator_pools.json")
    ap.add_argument("--out-root", type=str, required=True, help="Output root directory for repeats")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="Base seed; each repeat uses seed+i")
    ap.add_argument("--with-aos", dest="aos_enable", action="store_true", help="Force enable AOS via CLI flag")
    ap.add_argument("--aos-strict", action="store_true", help="Fail fast if AOS errors occur")
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra arg forwarded to GA; can repeat")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    per_run: List[Dict[str, float]] = []
    failed = 0
    for i in range(args.repeats):
        out_dir = os.path.join(args.out_root, f"run_{i+1}")
        seed = args.seed + i
        rc = run_one(args.config_dir, out_dir, seed, args)
        if rc != 0:
            print(f"[WARN] repeat {i+1} exited with code {rc}")
            failed += 1
            continue
        best_path = os.path.join(out_dir, "best_solution.json")
        evo_csv = os.path.join(out_dir, "evolution_log.csv")
        try:
            bf, acc, sel = parse_best(best_path)
            earliest, avg_div = parse_evolution(evo_csv)
            per_run.append({
                "best_fitness": bf,
                "cv_mean": acc,
                "num_selected": float(sel),
                "earliest_max_gen": earliest,
                "avg_diversity": avg_div,
            })
        except Exception as e:
            print(f"[WARN] failed to parse outputs for run_{i+1}: {e}")
            failed += 1

    # Aggregate
    def _mean(key: str) -> float:
        vals = [r[key] for r in per_run if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    avg_fitness = _mean("best_fitness")
    avg_accuracy = _mean("cv_mean")
    selected_num = _mean("num_selected")
    avg_max_fitness_step = _mean("earliest_max_gen")
    avg_diversity = _mean("avg_diversity")
    best_fitness = max([r["best_fitness"] for r in per_run], default=float("nan"))
    best_accuracy = max([r["cv_mean"] for r in per_run], default=float("nan"))

    summary = {
        "Avg_fitness": avg_fitness,
        "Avg_accuracy": avg_accuracy,
        "Selected_num": selected_num,
        "Avg_max_fitness_step": avg_max_fitness_step,
        "Avg_diversity": avg_diversity,
        "Best_Fitness": best_fitness,
        "Best_Accuracy": best_accuracy,
        "Repeats": len(per_run),
        "Failed": failed,
    }

    # Write CSV
    summary_csv = os.path.join(args.out_root, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print("[SUMMARY]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.6f}")
        else:
            print(f"- {k}: {v}")
    print(f"[OK] Wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()

