#!/usr/bin/env python3
"""
Run GA feature selection across a suite of dataset-specific config directories.

Each subfolder in --suite-dir is expected to contain:
  - task_info.json
  - algo_config.json
  - operator_pools.json

For each subfolder <name>, results are written to --out-root/<name>.

Usage example:
  python scripts/run_suite_configs.py \
    --suite-dir config_suites/fs_suite \
    --out-root results \
    --seed 42 \
    --generations 30 \
    --pop-size 60 \
    --op-switch-interval 5 \
    --with-aos --aos-strict

Notes:
  - Uses --use-config so that algo_config/operator_pools are respected per dataset.
  - You can pass multiple --extra-arg flags to forward any GA flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import List, Dict


def list_config_dirs(suite_dir: str) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(suite_dir)):
        path = os.path.join(suite_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("_"):
            continue
        # must have task_info.json
        if not os.path.exists(os.path.join(path, "task_info.json")):
            continue
        out.append(path)
    return out


def run_one(config_dir: str, out_root: str, args: argparse.Namespace) -> int:
    name = os.path.basename(config_dir.rstrip("/"))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "feature_selection_ga.py", "--use-config", "--config-dir", config_dir, "--output", out_dir]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.generations is not None:
        cmd += ["--generations", str(args.generations)]
    if args.pop_size is not None:
        cmd += ["--pop-size", str(args.pop_size)]
    if args.op_switch_interval is not None:
        cmd += ["--op-switch-interval", str(args.op_switch_interval)]
    if args.aos_enable:
        cmd.append("--aos-enable")
    if args.aos_strict:
        cmd.append("--aos-strict")
    for ea in (args.extra_arg or []):
        cmd.append(ea)
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)


def collect_best(out_root: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not os.path.exists(out_root):
        return rows
    for name in sorted(os.listdir(out_root)):
        path = os.path.join(out_root, name)
        if not os.path.isdir(path):
            continue
        best = os.path.join(path, "best_solution.json")
        if not os.path.exists(best):
            continue
        try:
            with open(best, "r") as f:
                data = json.load(f)
            rows.append({
                "dataset": name,
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


def write_summary(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("[WARN] No results collected, summary not written")
        return
    keys = [
        "dataset", "best_fitness", "cv_mean", "cv_std",
        "num_selected", "total_features", "selected_fraction",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] Summary written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=str, default="config_suites/default", help="Directory holding dataset config subfolders")
    ap.add_argument("--out-root", type=str, default="results", help="Root directory for outputs")
    ap.add_argument("--seed", type=int, default=None, help="Override seed (optional)")
    ap.add_argument("--generations", type=int, default=None, help="Override generations (optional)")
    ap.add_argument("--pop-size", type=int, default=None, help="Override pop size (optional)")
    ap.add_argument("--op-switch-interval", type=int, default=None, help="Override op switch interval (optional)")
    ap.add_argument("--with-aos", dest="aos_enable", action="store_true", help="Force enable AOS via CLI flag")
    ap.add_argument("--aos-strict", action="store_true", help="Force strict AOS via CLI flag")
    ap.set_defaults(aos_enable=False)
    ap.add_argument("--extra-arg", action="append", default=[], help="Extra arg forwarded to the GA CLI; can be repeated")
    args = ap.parse_args()

    cfg_dirs = list_config_dirs(args.suite_dir)
    if not cfg_dirs:
        print(f"[ERROR] No dataset config dirs found under {args.suite_dir}")
        raise SystemExit(1)
    print(f"[INFO] Found {len(cfg_dirs)} dataset configs under {args.suite_dir}")

    failed = 0
    for cdir in cfg_dirs:
        rc = run_one(cdir, args.out_root, args)
        if rc != 0:
            print(f"[WARN] {cdir} exited with code {rc}")
            failed += 1

    rows = collect_best(args.out_root)
    write_summary(os.path.join(args.out_root, "summary.csv"), rows)
    if failed:
        print(f"[DONE] Completed with {failed} failures.")
    else:
        print("[DONE] All dataset runs completed.")


if __name__ == "__main__":
    main()
