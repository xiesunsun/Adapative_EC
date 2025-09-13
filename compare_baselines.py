#!/usr/bin/env python3
"""
Compare GA feature selection against common baseline selectors under the
same evaluation protocol (CV + scoring + sparsity penalty alpha).

Baselines included:
 - all: use all features
 - random: random masks sampled M times
 - kbest_f: SelectKBest(f_classif) over a grid of k
 - kbest_mi: SelectKBest(mutual_info_classif) over a grid of k
 - rf_topk: top-k by RandomForest feature importances over a grid of k
 - l1_logistic: SelectFromModel with L1-penalized LogisticRegression over a small C grid

Outputs a results directory with a CSV summary and per-method JSON best subset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from feature_selection_ga import (
    load_dataset,
    make_classifier,
    evaluate_individual,
    ensure_valid,
    run_ga,
    cv_scores_for_mask,
)


def eval_mask_with_scores(mask: List[int], X, y, clf_name: str, scoring: str, cv: int, alpha: float, seed: Optional[int]) -> Tuple[float, float, float]:
    """Return (fitness, cv_mean, cv_std) for a mask."""
    rng_np = np.random.RandomState(seed if seed is not None else None)
    clf = make_classifier(clf_name)
    (fit,) = evaluate_individual(mask, X, y, clf, scoring, cv, alpha, rng_np)
    cv_mean, cv_std, _folds = cv_scores_for_mask(mask, X, y, clf, scoring, cv, seed)
    return float(fit), float(cv_mean), float(cv_std)


def mask_from_indices(indices: Sequence[int], d: int) -> List[int]:
    mask = [0] * d
    for i in indices:
        if 0 <= i < d:
            mask[i] = 1
    return mask


def baseline_all(d: int) -> List[int]:
    return [1] * d


def baseline_random(d: int, iters: int, init_prob: float, seed: Optional[int]) -> List[List[int]]:
    rng = random.Random(seed)
    masks: List[List[int]] = []
    for _ in range(iters):
        m = [1 if rng.random() < init_prob else 0 for _ in range(d)]
        ensure_valid(m, rng)
        masks.append(m)
    return masks


def baseline_kbest(X, y, k_grid: Sequence[int], score_fn: str) -> List[List[int]]:
    d = X.shape[1]
    masks: List[List[int]] = []
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        if score_fn == "f":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif score_fn == "mi":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError("score_fn must be 'f' or 'mi'")
        selector.fit(X, y)
        indices = np.argsort(selector.scores_)[-k:]
        masks.append(mask_from_indices(indices, d))
    return masks


def baseline_chi2_kbest(X, y, k_grid: Sequence[int]) -> List[List[int]]:
    """Chi-squared test with SelectKBest. Requires non-negative features; scale to [0,1]."""
    d = X.shape[1]
    X_pos = MinMaxScaler().fit_transform(X)
    masks: List[List[int]] = []
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        try:
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X_pos, y)
            indices = np.argsort(selector.scores_)[-k:]
        except Exception:
            indices = []
        masks.append(mask_from_indices(indices, d))
    return masks


def baseline_rf_topk(X, y, k_grid: Sequence[int], seed: Optional[int]) -> List[List[int]]:
    d = X.shape[1]
    rf = RandomForestClassifier(n_estimators=400, random_state=seed)
    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)
    masks: List[List[int]] = []
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        idx = order[-k:]
        masks.append(mask_from_indices(idx, d))
    return masks


def baseline_l1_logistic(X, y, C_grid: Sequence[float], seed: Optional[int]) -> List[List[int]]:
    d = X.shape[1]
    masks: List[List[int]] = []
    for C in C_grid:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500, random_state=seed)
        sfm = SelectFromModel(lr, prefit=False)
        sfm.fit(X, y)
        support = sfm.get_support()
        # If empty, keep the strongest coefficient feature
        if not np.any(support):
            try:
                lr.fit(X, y)
                coefs = np.abs(lr.coef_) if lr.coef_.ndim > 1 else np.abs(lr.coef_[None, :])
                strongest = int(np.argmax(coefs.sum(axis=0)))
                masks.append(mask_from_indices([strongest], d))
            except Exception:
                masks.append(mask_from_indices([], d))
        else:
            indices = np.where(support)[0]
            masks.append(mask_from_indices(indices, d))
    return masks


def baseline_rfe_svm(X, y, k_grid: Sequence[int], seed: Optional[int]) -> List[List[int]]:
    d = X.shape[1]
    masks: List[List[int]] = []
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        try:
            est = LinearSVC(C=1.0, dual=False, max_iter=3000, random_state=seed)
            rfe = RFE(estimator=est, n_features_to_select=k, step=1)
            rfe.fit(X, y)
            support = rfe.support_
            indices = np.where(support)[0]
        except Exception:
            indices = []
        masks.append(mask_from_indices(indices, d))
    return masks


def baseline_pca_importance(X, k_grid: Sequence[int]) -> List[List[int]]:
    """Select top-k features by PCA loading importance (unsupervised).
    Score per feature = sum_c |loading_{feature,c}| * explained_variance_ratio[c] over all components.
    """
    d = X.shape[1]
    masks: List[List[int]] = []
    try:
        pca = PCA(n_components=min(d, X.shape[0]))
        Xc = X - X.mean(axis=0, keepdims=True)
        pca.fit(Xc)
        loadings = pca.components_.T  # (d, n_comp)
        weights = pca.explained_variance_ratio_
        import numpy as _np
        scores = _np.abs(loadings) * weights[None, :]
        feat_score = scores.sum(axis=1)
        order = _np.argsort(feat_score)
    except Exception:
        # Fallback to variance ranking
        import numpy as _np
        order = _np.argsort(X.var(axis=0))
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        idx = order[-k:]
        masks.append(mask_from_indices(idx, d))
    return masks


def baseline_rfe(X, y, k_grid: Sequence[int], estimator) -> List[List[int]]:
    d = X.shape[1]
    masks: List[List[int]] = []
    for k in k_grid:
        if k < 1:
            continue
        k = min(k, d)
        try:
            rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
            rfe.fit(X, y)
            support = rfe.support_
            indices = np.where(support)[0]
        except Exception:
            indices = []
        masks.append(mask_from_indices(indices, d))
    return masks


def default_k_grid(d: int, max_frac: float = 0.5) -> List[int]:
    """Return a small grid of k values up to a fraction of d (default 50%).
    Includes 1 and roughly spaced fractions; avoids k=d by default to reduce trivial select-all picks.
    """
    try:
        mf = float(max_frac)
    except Exception:
        mf = 0.5
    mf = 1.0 if mf > 1.0 else (0.0 if mf < 0.0 else mf)
    max_k = max(1, int(round(d * mf)))
    base = [1, max(2, d // 10), d // 5, d // 3, d // 2, d]
    grid = sorted(set(k for k in base if 1 <= k <= max_k))
    if max_k not in grid:
        grid.append(max_k)
    return sorted(set(grid))


def main():
    ap = argparse.ArgumentParser(description="Compare GA vs baseline feature selectors")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str)
    src.add_argument("--sklearn-dataset", type=str)
    src.add_argument("--openml-name", type=str)
    src.add_argument("--openml-id", type=str)
    ap.add_argument("--target-col", type=str)
    ap.add_argument("--openml-version", type=int, default=None)
    ap.add_argument("--data-home", type=str, default=None)

    ap.add_argument("--classifier", type=str, default="logistic", choices=["logistic", "svm", "rf"])
    ap.add_argument("--scoring", type=str, default="accuracy")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--baselines", type=str, default="rfe_svm,lasso,chi2,mi,rf_importance,pca",
                    help="Comma-separated baselines to run. Supported: rfe_svm, lasso, chi2, mi, rf_importance, pca")
    ap.add_argument("--random-iters", type=int, default=200)
    ap.add_argument("--init-prob", type=float, default=0.1)
    ap.add_argument("--k-grid", type=str, default="auto", help="Comma ints or 'auto'")
    ap.add_argument("--k-max-frac", type=float, default=0.5, help="When k-grid='auto', cap k to floor(d*frac). Default 0.5")
    ap.add_argument("--C-grid", type=str, default="0.01,0.1,1,10")

    # Optionally include GA run for side-by-side comparison
    ap.add_argument("--include-ga", action="store_true", help="Also run GA and include in summary")
    ap.add_argument("--pop-size", type=int, default=60)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--cxpb", type=float, default=0.6)
    ap.add_argument("--mutpb", type=float, default=0.2)

    ap.add_argument("--output", type=str, default="baseline_results")
    args = ap.parse_args()

    data = load_dataset(
        csv=args.csv,
        target_col=args.target_col,
        sklearn_dataset=args.sklearn_dataset,
        openml_name=args.openml_name,
        openml_id=args.openml_id,
        openml_version=args.openml_version,
        data_home=args.data_home,
    )

    X, y, feat_names = data.X, data.y, [str(n) for n in data.feature_names]
    d = X.shape[1]

    if args.k_grid == "auto":
        k_grid = default_k_grid(d, max_frac=args.k_max_frac)
    else:
        k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]

    C_grid = [float(x) for x in args.C_grid.split(",") if x.strip()]
    methods = [m.strip() for m in args.baselines.split(",") if m.strip()]

    os.makedirs(args.output, exist_ok=True)
    rows: List[Dict] = []

    for method in methods:
        best_fit = -1e12
        best_mask: Optional[List[int]] = None
        detail: Dict[str, object] = {"method": method}

        if method == "rfe_svm":
            masks = baseline_rfe_svm(X, y, k_grid, args.seed)
            detail["params"] = {"k_grid": k_grid, "estimator": "LinearSVC"}
        elif method == "lasso":
            masks = baseline_l1_logistic(X, y, C_grid, args.seed)
            detail["params"] = {"C_grid": C_grid}
        elif method == "chi2":
            masks = baseline_chi2_kbest(X, y, k_grid)
            detail["params"] = {"k_grid": k_grid}
        elif method == "mi":
            masks = baseline_kbest(X, y, k_grid, score_fn="mi")
            detail["params"] = {"k_grid": k_grid}
        elif method == "rf_importance":
            masks = baseline_rf_topk(X, y, k_grid, args.seed)
            detail["params"] = {"k_grid": k_grid}
        elif method == "pca":
            masks = baseline_pca_importance(X, k_grid)
            detail["params"] = {"k_grid": k_grid}
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        # Evaluate all masks; tie-break by fewer features when fitness nearly equal
        best_cv_mean = float("nan")
        best_cv_std = float("nan")
        best_k = d + 1
        for m in masks:
            fit, cv_mean, cv_std = eval_mask_with_scores(m, X, y, args.classifier, args.scoring, args.cv, args.alpha, args.seed)
            k_now = int(sum(1 for b in m if b))
            if (fit > best_fit + 1e-12) or (abs(fit - best_fit) <= 1e-12 and k_now < best_k):
                best_fit = fit
                best_cv_mean = cv_mean
                best_cv_std = cv_std
                best_mask = m
                best_k = k_now

        if best_mask is None:
            best_mask = [0] * d

        sel_idx = [i for i, b in enumerate(best_mask) if b == 1]
        sel_names = [str(feat_names[i]) for i in sel_idx]

        rows.append({
            "method": method,
            "best_fitness": best_fit,
            "cv_mean": best_cv_mean,
            "cv_std": best_cv_std,
            "num_selected": len(sel_idx),
            "total_features": d,
            "selected_indices": sel_idx,
            "selected_features": sel_names,
            "params": json.dumps(detail.get("params", {})),
        })

        with open(os.path.join(args.output, f"best_{method}.json"), "w") as f:
            json.dump({
                "method": method,
                "best_fitness": best_fit,
                "cv_mean": best_cv_mean,
                "cv_std": best_cv_std,
                "selected_indices": sel_idx,
                "selected_features": sel_names,
                "num_selected": len(sel_idx),
                "total_features": d,
                "params": detail.get("params", {}),
            }, f, indent=2)

    df = pd.DataFrame(rows)
    df.sort_values(by="best_fitness", ascending=False, inplace=True)
    df.to_csv(os.path.join(args.output, "baseline_summary.csv"), index=False)
    print(f"Saved baseline comparison to '{args.output}/baseline_summary.csv'")

    if args.include_ga:
        # Run GA with provided hyperparameters and append to the same CSV
        best_mask, best_fit, _log, _hof = run_ga(
            X=X,
            y=y,
            feature_names=feat_names,
            classifier=args.classifier,
            scoring=args.scoring,
            cv=args.cv,
            alpha=args.alpha,
            pop_size=args.pop_size,
            generations=args.generations,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            init_prob=args.init_prob,
            seed=args.seed,
        )
        sel_idx = [i for i, b in enumerate(best_mask) if b == 1]
        sel_names = [str(feat_names[i]) for i in sel_idx]
        # Compute unpenalized CV for GA best mask
        clf_ga = make_classifier(args.classifier)
        ga_cv_mean, ga_cv_std, _folds = cv_scores_for_mask(best_mask, X, y, clf_ga, args.scoring, args.cv, args.seed)
        ga_row = {
            "method": "ga",
            "best_fitness": best_fit,
            "cv_mean": ga_cv_mean,
            "cv_std": ga_cv_std,
            "num_selected": len(sel_idx),
            "total_features": d,
            "selected_indices": sel_idx,
            "selected_features": sel_names,
            "params": json.dumps({
                "pop_size": args.pop_size,
                "generations": args.generations,
                "cxpb": args.cxpb,
                "mutpb": args.mutpb,
                "init_prob": args.init_prob,
            }),
        }
        # Save GA json
        with open(os.path.join(args.output, "best_ga.json"), "w") as f:
            json.dump({**ga_row, "selected_features": sel_names}, f, indent=2)

        # Append to CSV
        df2 = pd.read_csv(os.path.join(args.output, "baseline_summary.csv"))
        df2 = pd.concat([df2, pd.DataFrame([ga_row])], ignore_index=True)
        df2.sort_values(by="best_fitness", ascending=False, inplace=True)
        df2.to_csv(os.path.join(args.output, "baseline_summary.csv"), index=False)
        print("Appended GA results to baseline_summary.csv")


if __name__ == "__main__":
    main()
