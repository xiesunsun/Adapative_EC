#!/usr/bin/env python3
"""
Run baseline feature selection methods multiple times and aggregate results.

Aggregated metrics per method (across repeats):
- Avg_fitness: mean(best_fitness)
- Avg_accuracy: mean(cv_mean)
- Selected_num: mean(num_selected)
- Avg_max_fitness_step: NaN (not applicable for baselines without generations)
- Avg_diversity: NaN (not applicable)
- Best_fitness: max(best_fitness)
- Best_Accuracy: max(cv_mean)

Usage examples:
  # Breast cancer, SVM, 3 repeats
  python scripts/run_baselines_repeats.py \
    --sklearn-dataset breast_cancer \
    --classifier svm \
    --repeats 3 \
    --seed 42 \
    --out-root baseline_repeats/bc

Notes:
- Wraps compare_baselines.py. Each repeat writes to <out-root>/run_i and an aggregated summary is saved to <out-root>/summary.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

import pandas as pd


def run_once(args: argparse.Namespace, run_idx: int, seed: int) -> int:
    out_dir = os.path.join(args.out_root, f"run_{run_idx}")
    os.makedirs(out_dir, exist_ok=True)
    cmd: List[str] = [
        sys.executable, "compare_baselines.py",
        "--classifier", args.classifier,
        "--scoring", args.scoring,
        "--cv", str(args.cv),
        "--alpha", str(args.alpha),
        "--seed", str(seed),
        "--output", out_dir,
    ]
    if args.sklearn_dataset:
        cmd += ["--sklearn-dataset", args.sklearn_dataset]
    elif args.csv and args.target_col:
        cmd += ["--csv", args.csv, "--target-col", args.target_col]
    elif args.openml_name or args.openml_id:
        if args.openml_name:
            cmd += ["--openml-name", args.openml_name]
        if args.openml_id:
            cmd += ["--openml-id", str(args.openml_id)]
        if args.openml_version is not None:
            cmd += ["--openml-version", str(args.openml_version)]
    if args.k_grid:
        cmd += ["--k-grid", args.k_grid]
    if args.C_grid:
        cmd += ["--C-grid", args.C_grid]
    if args.baselines:
        cmd += ["--baselines", args.baselines]
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)


def aggregate(out_root: str, repeats: int, summary_name: str = "summary.csv") -> None:
    frames: List[pd.DataFrame] = []
    for i in range(1, repeats + 1):
        p = os.path.join(out_root, f"run_{i}", "baseline_summary.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["run"] = i
        frames.append(df)
    if not frames:
        print("[WARN] No per-run baseline_summary.csv files found; nothing to aggregate.")
        return
    df_all = pd.concat(frames, ignore_index=True)
    # Compute aggregates per method
    # Note: baselines have no per-generation logs; mark Avg_max_fitness_step/Avg_diversity as NaN
    agg = df_all.groupby("method").agg({
        "best_fitness": ["mean", "max"],
        "cv_mean": ["mean", "max"],
        "num_selected": "mean",
    }).reset_index()
    agg.columns = [
        "method",
        "Avg_fitness", "Best_fitness",
        "Avg_accuracy", "Best_Accuracy",
        "Selected_num",
    ]
    agg["Avg_max_fitness_step"] = float("nan")
    agg["Avg_diversity"] = float("nan")
    # Reorder columns
    agg = agg[[
        "method",
        "Avg_fitness", "Avg_accuracy", "Selected_num",
        "Avg_max_fitness_step", "Avg_diversity",
        "Best_fitness", "Best_Accuracy",
    ]]
    out_csv = os.path.join(out_root, summary_name)
    agg.to_csv(out_csv, index=False)
    print(f"[OK] Aggregated baseline summary -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    # dataset sources (choose one; default sklearn breast_cancer)
    ap.add_argument("--sklearn-dataset", type=str, default="breast_cancer")
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--target-col", type=str, default=None)
    ap.add_argument("--openml-name", type=str, default=None)
    ap.add_argument("--openml-id", type=str, default=None)
    ap.add_argument("--openml-version", type=int, default=None)
    # evaluation settings
    ap.add_argument("--classifier", type=str, default="svm")
    ap.add_argument("--scoring", type=str, default="accuracy")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.02)
    # baselines config (align with compare_baselines.py defaults)
    ap.add_argument("--baselines", type=str, default="rfe_svm,lasso,chi2,mi,rf_importance,pca")
    ap.add_argument("--k-grid", type=str, default="auto")
    ap.add_argument("--C-grid", type=str, default="0.01,0.1,1,10")
    # repeats
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="base seed; repeat i uses seed+i")
    # output
    ap.add_argument("--out-root", type=str, default="baseline_repeats/bc")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    failed = 0
    for i in range(1, args.repeats + 1):
        seed = args.seed + (i - 1)
        rc = run_once(args, i, seed)
        if rc != 0:
            print(f"[WARN] run_{i} exited with code {rc}")
            failed += 1

    aggregate(args.out_root, args.repeats)
    if failed:
        print(f"[DONE] Completed with {failed} failures.")
    else:
        print("[DONE] All baseline repeats completed.")


if __name__ == "__main__":
    main()
