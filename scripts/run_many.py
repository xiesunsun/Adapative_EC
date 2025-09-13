#!/usr/bin/env python3
"""
Batch runner for GA feature selection across multiple datasets.

Supports:
- sklearn datasets: iris, wine, breast_cancer
- synthetic CSV datasets generated on the fly with varying feature counts

Writes each run into a separate subdirectory and produces a summary CSV.

Examples:
  python scripts/run_many.py --suite default --out ga_results/batch --seed 42 \
    --generations 30 --pop-size 60 --op-switch-interval 10 --aos

Notes:
- Uses --use-config so AOS endpoint/model/debug etc. come from config/algo_config.json.
- Override per-run data source via CLI (sklearn/CSV), leaving the rest to config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Task:
    name: str
    kind: str  # "sklearn" | "csv"
    sklearn_name: Optional[str] = None
    csv_path: Optional[str] = None
    target_col: Optional[str] = None


def make_default_suite(out_dir: str) -> Tuple[List[Task], List[Task]]:
    """Return (sklearn_tasks, synthetic_csv_tasks)."""
    sklearn_tasks = [
        Task(name="iris", kind="sklearn", sklearn_name="iris"),
        Task(name="wine", kind="sklearn", sklearn_name="wine"),
        Task(name="breast_cancer", kind="sklearn", sklearn_name="breast_cancer"),
    ]
    # Prepare synthetic datasets with increasing feature counts
    synth_feature_counts = [100, 500]
    synth_tasks: List[Task] = []
    data_dir = os.path.join(out_dir, "_data")
    os.makedirs(data_dir, exist_ok=True)
    for nf in synth_feature_counts:
        path = os.path.join(data_dir, f"synth_nf{nf}.csv")
        synth_tasks.append(Task(name=f"synth_nf{nf}", kind="csv", csv_path=path, target_col="target"))
    return sklearn_tasks, synth_tasks


def ensure_synthetic_csv(path: str, n_samples: int, n_features: int, random_state: int = 42) -> None:
    from sklearn.datasets import make_classification
    rs = random_state
    n_informative = max(2, int(0.1 * n_features))
    n_redundant = max(0, int(0.05 * n_features))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        shuffle=True,
        random_state=rs,
    )
    # Write CSV with header: f0...f{n-1}, target
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(X.shape[1])] + ["target"]
        writer.writerow(header)
        for i in range(X.shape[0]):
            writer.writerow(list(map(float, X[i].tolist())) + [int(y[i])])


def run_task(
    task: Task,
    out_root: str,
    seed: int,
    generations: int,
    pop_size: int,
    op_switch_interval: int,
    aos: bool,
    extra_args: Optional[List[str]] = None,
) -> int:
    cmd = [sys.executable, "feature_selection_ga.py", "--use-config"]
    out_dir = os.path.join(out_root, task.name)
    os.makedirs(out_dir, exist_ok=True)
    if task.kind == "sklearn":
        cmd += ["--sklearn-dataset", task.sklearn_name or ""]
    elif task.kind == "csv":
        assert task.csv_path and task.target_col
        cmd += ["--csv", task.csv_path, "--target-col", task.target_col]
    cmd += [
        "--output", out_dir,
        "--seed", str(seed),
        "--generations", str(generations),
        "--pop-size", str(pop_size),
        "--op-switch-interval", str(op_switch_interval),
    ]
    if aos:
        cmd += ["--aos-enable"]
    if extra_args:
        cmd += extra_args
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)


def collect_best(out_root: str, tasks: List[Task]) -> List[dict]:
    rows: List[dict] = []
    for t in tasks:
        out_dir = os.path.join(out_root, t.name)
        best_path = os.path.join(out_dir, "best_solution.json")
        if not os.path.exists(best_path):
            continue
        try:
            with open(best_path, "r") as f:
                data = json.load(f)
            rows.append({
                "task": t.name,
                "best_fitness": data.get("best_fitness"),
                "cv_mean": data.get("cv_mean"),
                "cv_std": data.get("cv_std"),
                "num_selected": data.get("num_selected"),
                "total_features": data.get("total_features"),
                "selected_fraction": data.get("selected_fraction"),
            })
        except Exception:
            pass
    return rows


def write_summary_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = [
        "task", "best_fitness", "cv_mean", "cv_std",
        "num_selected", "total_features", "selected_fraction",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Summary written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="default", help="Suite name (default)")
    ap.add_argument("--out", type=str, default="ga_results/batch", help="Output root directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--pop-size", type=int, default=60)
    ap.add_argument("--op-switch-interval", type=int, default=10)
    ap.add_argument("--no-aos", dest="aos", action="store_false", help="Disable AOS for the batch runs (override config)")
    ap.add_argument("--with-aos", dest="aos", action="store_true", help="Enable AOS for the batch runs (override config)")
    ap.set_defaults(aos=True)
    ap.add_argument("--synthetic-samples", type=int, default=600, help="Samples for synthetic datasets")
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra arg to pass to GA (can repeat)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    skl_tasks, synth_tasks = make_default_suite(args.out)
    # Create synthetic CSV files
    for t in synth_tasks:
        nf = int(t.name.split("nf")[-1])
        ensure_synthetic_csv(t.csv_path or "", n_samples=args.synthetic_samples, n_features=nf, random_state=args.seed)
        print(f"[DATA] Wrote synthetic CSV: {t.csv_path}")

    # Run sklearn tasks
    failed = 0
    for t in skl_tasks + synth_tasks:
        rc = run_task(
            t,
            out_root=args.out,
            seed=args.seed,
            generations=args.generations,
            pop_size=args.pop_size,
            op_switch_interval=args.op_switch_interval,
            aos=args.aos,
            extra_args=args.extra_arg,
        )
        if rc != 0:
            print(f"[WARN] Task {t.name} exited with code {rc}")
            failed += 1

    # Collect summary
    rows = collect_best(args.out, skl_tasks + synth_tasks)
    write_summary_csv(os.path.join(args.out, "summary.csv"), rows)
    if failed:
        print(f"[DONE] Completed with {failed} failures.")
    else:
        print("[DONE] All tasks completed.")


if __name__ == "__main__":
    main()

