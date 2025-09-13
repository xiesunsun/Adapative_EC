#!/usr/bin/env python3
"""
Generate a suite of dataset-specific config directories that keep the same
algo_config.json and operator_pools.json, while varying only task_info.json.

Each dataset gets its own folder under --out, containing:
  - task_info.json (generated)
  - operator_pools.json (copied from --base)
  - algo_config.json (copied from --base)

You can include:
  - sklearn datasets: iris, wine, breast_cancer
  - OpenML datasets by name or id (requires network at run time)
  - CSV datasets by path:target_col
  - Synthetic CSV datasets with varying feature counts (created under out/_data)

Example:
  python scripts/make_suite_configs.py \
    --base config \
    --out config_suites/default \
    --sklearn iris wine breast_cancer \
    --synth 100 500 --synth-samples 600

Then run each dataset with:
  python feature_selection_ga.py --use-config --config-dir config_suites/default/iris --output ga_results/iris
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from typing import List, Tuple


def _copy_base(base_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for fname in ("operator_pools.json", "algo_config.json"):
        src = os.path.join(base_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _task_info_sklearn(name: str) -> dict:
    pretty = name if name in ("iris", "wine", "breast_cancer") else f"sklearn:{name}"
    return {
        "dataset_name": f"sklearn:{name}",
        "dataset_source": "sklearn",
        "openml": {"name": None, "id": None, "version": None},
        "csv": {"path": None, "target_col": None},
        "dataset_size": "unknown",
        "num_features": 0,
        "classification_model": "logistic",
        "scoring": "accuracy",
        "cv_folds": 5,
        "fitness_function": "CV_mean - alpha * (k/d)",
        "alpha": 0.02,
        "notes": f"Sklearn dataset: {pretty}"
    }


def _task_info_openml_by_name(name: str) -> dict:
    return {
        "dataset_name": f"openml:{name}",
        "dataset_source": "openml",
        "openml": {"name": name, "id": None, "version": None},
        "csv": {"path": None, "target_col": None},
        "dataset_size": "unknown",
        "num_features": 0,
        "classification_model": "logistic",
        "scoring": "accuracy",
        "cv_folds": 5,
        "fitness_function": "CV_mean - alpha * (k/d)",
        "alpha": 0.02,
        "notes": f"OpenML dataset by name: {name}"
    }


def _task_info_openml_by_id(oid: int, version: int | None) -> dict:
    return {
        "dataset_name": f"openml:{oid}",
        "dataset_source": "openml",
        "openml": {"name": None, "id": oid, "version": version},
        "csv": {"path": None, "target_col": None},
        "dataset_size": "unknown",
        "num_features": 0,
        "classification_model": "logistic",
        "scoring": "accuracy",
        "cv_folds": 5,
        "fitness_function": "CV_mean - alpha * (k/d)",
        "alpha": 0.02,
        "notes": f"OpenML dataset by id: {oid}"
    }


def _task_info_csv(path: str, target: str) -> dict:
    return {
        "dataset_name": f"csv:{os.path.basename(path)}",
        "dataset_source": "csv",
        "openml": {"name": None, "id": None, "version": None},
        "csv": {"path": path, "target_col": target},
        "dataset_size": "unknown",
        "num_features": 0,
        "classification_model": "logistic",
        "scoring": "accuracy",
        "cv_folds": 5,
        "fitness_function": "CV_mean - alpha * (k/d)",
        "alpha": 0.02,
        "notes": f"CSV dataset: {path}"
    }


def _ensure_synth_csv(path: str, n_samples: int, n_features: int, random_state: int = 42) -> None:
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, int(0.1 * n_features)),
        n_redundant=max(0, int(0.05 * n_features)),
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        shuffle=True,
        random_state=random_state,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(X.shape[1])] + ["target"]
        writer.writerow(header)
        for i in range(X.shape[0]):
            writer.writerow(list(map(float, X[i].tolist())) + [int(y[i])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="config", help="Base config dir (contains operator_pools.json & algo_config.json)")
    ap.add_argument("--out", type=str, default="config_suites/default", help="Output root for generated config dirs")
    ap.add_argument("--sklearn", nargs="*", default=["iris", "wine", "breast_cancer"], help="Sklearn datasets to include")
    ap.add_argument("--openml-name", nargs="*", default=[], help="OpenML dataset names to include")
    ap.add_argument("--openml-id", nargs="*", default=[], help="OpenML dataset IDs to include (ints)")
    ap.add_argument("--openml-version", type=int, default=None, help="Optional OpenML version for all IDs")
    ap.add_argument("--csv", nargs="*", default=[], help="CSV datasets specified as path:target_col")
    ap.add_argument("--synth", nargs="*", default=[], help="Synthetic datasets by feature counts, e.g., 100 500 1000")
    ap.add_argument("--synth-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.out, "_data")
    os.makedirs(data_dir, exist_ok=True)

    # Build task list as tuples: (folder_name, task_info_dict)
    tasks: List[Tuple[str, dict]] = []
    # Sklearn
    for name in args.sklearn:
        folder = name
        tasks.append((folder, _task_info_sklearn(name)))
    # OpenML by name
    for nm in args.openml_name:
        folder = f"openml_name_{nm}"
        tasks.append((folder, _task_info_openml_by_name(nm)))
    # OpenML by id
    for oid_s in args.openml_id:
        try:
            oid = int(oid_s)
        except Exception:
            continue
        folder = f"openml_id_{oid}"
        tasks.append((folder, _task_info_openml_by_id(oid, args.openml_version)))
    # CSV datasets
    for item in args.csv:
        if ":" not in item:
            continue
        path, target = item.split(":", 1)
        folder = f"csv_{os.path.splitext(os.path.basename(path))[0]}"
        tasks.append((folder, _task_info_csv(path, target)))
    # Synthetic datasets
    for nf_s in args.synth:
        try:
            nf = int(nf_s)
        except Exception:
            continue
        csv_path = os.path.join(data_dir, f"synth_nf{nf}.csv")
        _ensure_synth_csv(csv_path, n_features=nf, n_samples=args.synth_samples, random_state=args.seed)
        folder = f"synth_nf{nf}"
        tasks.append((folder, _task_info_csv(csv_path, "target")))

    # Materialize each config dir
    for folder, info in tasks:
        cdir = os.path.join(args.out, folder)
        _copy_base(args.base, cdir)
        _write_json(os.path.join(cdir, "task_info.json"), info)
        print(f"[OK] Wrote {cdir}")

    print(f"[DONE] Generated {len(tasks)} dataset config dirs under {args.out}")


if __name__ == "__main__":
    main()

