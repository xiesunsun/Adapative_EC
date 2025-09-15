#!/usr/bin/env python3
"""
Genetic Algorithm Feature Selection using DEAP.

Supports CSV input (with a target column) or built-in scikit-learn datasets.
Evaluates subsets via cross-validation using a chosen classifier, with an
optional sparsity penalty to prefer fewer features.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_openml
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Any, Dict, Union
from datetime import datetime, timezone

# --- Global worker evaluation context for multiprocessing ---
_EVAL_CTX: Dict[str, Any] = {}

def _init_eval_worker(ctx: Dict[str, Any]) -> None:
    """Initializer for worker processes: build per-worker classifier and store context.
    Avoids pickling large objects per task and re-creating the classifier each call.
    """
    global _EVAL_CTX
    # Build classifier once per worker
    try:
        local = dict(ctx)
        local["clf"] = make_classifier(ctx["classifier"])  # type: ignore
        # Build fixed CV splitter if requested
        try:
            if bool(local.get("fixed_cv", False)):
                n_splits = int(local.get("n_splits", 5))
                repeats = int(local.get("cv_repeats", 1))
                base_seed = int(local.get("base_seed", 0))
                if repeats > 1:
                    local["cv_obj"] = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=base_seed)
                else:
                    local["cv_obj"] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
        except Exception:
            pass
        _EVAL_CTX = local
    except Exception:
        _EVAL_CTX = dict(ctx)

def _seed_from_individual(ind: List[int], base_seed: int) -> int:
    # Stable seed derived from base_seed and the individual's bit pattern
    try:
        h = 0
        for b in ind:
            h = ((h << 1) ^ int(b)) & 0x7fffffff
        return int((base_seed * 1315423911) ^ h) & 0x7fffffff
    except Exception:
        return int(base_seed) & 0x7fffffff

def _evaluate_individual_picklable(individual: List[int]) -> Tuple[float]:
    """Picklable evaluate function that reads data from the global worker context."""
    global _EVAL_CTX
    X = _EVAL_CTX.get("X")
    y = _EVAL_CTX.get("y")
    scoring = _EVAL_CTX.get("scoring", "accuracy")
    cv_param = _EVAL_CTX.get("cv_obj", None)
    if cv_param is None:
        cv_param = int(_EVAL_CTX.get("cv", 5))
    alpha = float(_EVAL_CTX.get("alpha", 0.0))
    n_jobs_eval = int(_EVAL_CTX.get("n_jobs_eval", 1))
    # Prefer per-worker prebuilt classifier; fallback to building one
    clf = _EVAL_CTX.get("clf")
    if clf is None:
        clf = make_classifier(_EVAL_CTX.get("classifier", "logistic"))
    base_seed = int(_EVAL_CTX.get("base_seed", 0))
    # Derive a deterministic RNG for CV splitter seeds used when cv is int/randomized
    rng = np.random.RandomState(_seed_from_individual(individual, base_seed))
    return evaluate_individual(individual, X, y, clf, scoring, cv_param, alpha, rng, n_jobs_eval)

# Optional AOS imports (LLM-driven adaptive operator selection)
try:
    from aos.adapter import AOSAdapter
    from aos.config_loader import (
        load_configs,
        build_state_payload_from_configs,
        build_decision_payload_from_configs,
    )
    from aos.prompts import build_state_prompt_v2, build_decision_prompt_v2
    from aos.schema import validate_decision
except Exception as _AOS_IMPORT_EXC:  # capture reason for diagnostics
    AOSAdapter = None  # type: ignore
    AOS_IMPORT_ERROR = str(_AOS_IMPORT_EXC)
else:
    AOS_IMPORT_ERROR = ""


# -----------------------------
# Data loading utilities
# -----------------------------


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]


def load_dataset(
    csv: Optional[str],
    target_col: Optional[str],
    sklearn_dataset: Optional[str],
    openml_name: Optional[str],
    openml_id: Optional[str],
    openml_version: Optional[int],
    data_home: Optional[str],
) -> Dataset:
    if csv:
        if not target_col:
            raise ValueError("--target-col is required when using --csv")
        df = pd.read_csv(csv)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = [c for c in df.columns if c != target_col]
        return Dataset(X=X, y=y, feature_names=feature_names)

    if sklearn_dataset:
        name = sklearn_dataset.lower()
        if name == "breast_cancer":
            data = load_breast_cancer()
        elif name == "iris":
            data = load_iris()
        elif name == "wine":
            data = load_wine()
        else:
            raise ValueError("Unsupported sklearn dataset. Choose breast_cancer|iris|wine")
        X = data.data
        y = data.target
        feature_names = list(data.feature_names) if hasattr(data, "feature_names") else [f"f{i}" for i in range(X.shape[1])]
        return Dataset(X=X, y=y, feature_names=feature_names)

    if openml_name or openml_id:
        # Fetch from OpenML (requires network). Uses sklearn's cache if data_home provided.
        fetch_kwargs = {"as_frame": True}
        # Silence sklearn fetch_openml future warning by using the new default explicitly
        try:
            fetch_kwargs["parser"] = "auto"
        except Exception:
            pass
        if data_home:
            fetch_kwargs["data_home"] = data_home
        if openml_version is not None:
            fetch_kwargs["version"] = openml_version

        if openml_name:
            ds = fetch_openml(name=openml_name, **fetch_kwargs)
        else:
            # openml_id may be str; cast to int if possible
            try:
                data_id = int(openml_id) if openml_id is not None else None
            except ValueError:
                raise ValueError("--openml-id must be an integer")
            if data_id is None:
                raise ValueError("Provide either --openml-name or --openml-id")
            ds = fetch_openml(data_id=data_id, **fetch_kwargs)

        if target_col:
            if ds.frame is None:
                raise ValueError("OpenML fetch did not return a frame; cannot use --target-col override")
            if target_col not in ds.frame.columns:
                raise ValueError(f"Target column '{target_col}' not found in OpenML dataset")
            y_series = ds.frame[target_col]
            X_df = ds.frame.drop(columns=[target_col])
        else:
            X_df = ds.data
            y_series = ds.target

        # Convert to numpy, handling categoricals and strings if present
        y = y_series
        if hasattr(y, "dtype") and (y.dtype.kind in ("O", "U", "S")):
            y = pd.factorize(y_series)[0]
        elif str(getattr(y, "dtype", "")) == "category":
            y = y_series.cat.codes.values
        else:
            y = np.asarray(y_series)

        feature_names = list(X_df.columns)
        X = X_df.values
        return Dataset(X=X, y=y, feature_names=feature_names)

    raise ValueError("Provide either --csv with --target-col or --sklearn-dataset")


# -----------------------------
# Classifier factory
# -----------------------------


def make_classifier(name: str) -> Pipeline:
    name = name.lower()
    if name == "logistic":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, solver="liblinear")),
        ])
    elif name == "svm":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", gamma="scale")),
        ])
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
    else:
        raise ValueError("Unsupported classifier. Choose logistic|svm|rf")
    return clf


# -----------------------------
# GA Feature Selection
# -----------------------------


def evaluate_individual(
    individual: List[int],
    X: np.ndarray,
    y: np.ndarray,
    clf: Pipeline,
    scoring: str,
    cv: Union[int, object],
    alpha: float,
    rng: np.random.RandomState,
    n_jobs_eval: int = 1,
) -> Tuple[float]:
    mask = np.array(individual, dtype=bool)
    # Ensure at least one feature selected; if not, return a very low fitness
    if not mask.any():
        return (-1e6,)

    X_sel = X[:, mask]
    # CV splitter: allow either an int or a provided splitter object (fixed CV)
    if isinstance(cv, int):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(rng.randint(0, 2**31 - 1)))
    else:
        skf = cv
    try:
        scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=skf, n_jobs=n_jobs_eval)
        mean_score = float(np.mean(scores))
    except Exception:
        # In case model fails on a particular subset, assign very low fitness
        return (-1e6,)

    # Apply sparsity penalty proportional to fraction of selected features
    sparsity_penalty = alpha * (mask.sum() / mask.size)
    fitness = mean_score - sparsity_penalty
    return (fitness,)


def cv_scores_for_mask(
    individual: List[int],
    X: np.ndarray,
    y: np.ndarray,
    clf: Pipeline,
    scoring: str,
    cv: int,
    random_state: Optional[int],
    cv_override: Optional[object] = None,
) -> Tuple[float, float, List[float]]:
    """
    Compute unpenalized CV scores (mean, std, per-fold) for a given mask.
    Repairs empty masks by returning NaN/empty scores.
    """
    mask = np.array(individual, dtype=bool)
    if not mask.any():
        return float("nan"), float("nan"), []
    X_sel = X[:, mask]
    if cv_override is not None:
        skf = cv_override
    else:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=(random_state if random_state is not None else 0))
    try:
        scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=skf, n_jobs=None)
        return float(np.mean(scores)), float(np.std(scores)), [float(s) for s in scores]
    except Exception:
        return float("nan"), float("nan"), []


def ensure_valid(individual: List[int], rng: random.Random) -> None:
    # Repair: if no feature is selected, flip one random bit to 1
    if sum(individual) == 0:
        idx = rng.randrange(len(individual))
        individual[idx] = 1


def population_diversity(population: List[List[int]]) -> float:
    """
    Compute average pairwise Hamming distance fraction across the population.
    Returns a value in [0, 1], where 0 means identical individuals and 1 means
    every bit differs on average.
    Efficient computation using per-bit counts: diversity_frac =
        (1 / (L * N * (N - 1))) * sum_j 2 * n1_j * (N - n1_j)
    where L is genome length and n1_j is the count of ones at bit j.
    """
    if not population:
        return float("nan")
    N = len(population)
    L = len(population[0]) if population[0] is not None else 0
    if N < 2 or L == 0:
        return 0.0
    M = np.asarray(population, dtype=int)  # shape (N, L)
    n1 = M.sum(axis=0)  # ones per bit
    pairs_diff_per_bit = 2.0 * n1 * (N - n1)  # ordered pairs differing at bit j
    total_pairs_ordered = N * (N - 1)
    diversity_raw = float(pairs_diff_per_bit.sum()) / total_pairs_ordered  # avg Hamming count
    diversity_frac = diversity_raw / L
    return diversity_frac


def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    classifier: str,
    scoring: str,
    cv: int,
    alpha: float,
    pop_size: int,
    generations: int,
    cxpb: float,
    mutpb: float,
    init_prob: float,
    seed: Optional[int],
    n_procs: int = 1,
    n_jobs_eval: int = 1,
    fixed_cv: bool = False,
    cv_repeats: int = 1,
    op_switch_interval: int = 10,
    disable_op_switch: bool = False,
    sr_fair_cv: bool = False,
    adaptive_switch: bool = False,
    switch_base_interval: int = 10,
    switch_min_interval: int = 5,
    switch_max_interval: int = 40,
    switch_window: int = 5,
    switch_patience: int = 3,
    switch_ir_thresh: float = 0.0005,
    switch_sr_thresh: float = 0.05,
    switch_deltaD_thresh: float = 0.0,
    switch_cooldown: int = 1,
    output_dir: str = "ga_results",
    # LLM AOS parameters
    aos_enable: bool = False,
    aos_endpoint: str = "https://api.sunxie.xyz",
    aos_model: str = "gpt-4o-mini",
    aos_api_key: str = "",
    aos_timeout: float = 30.0,
    aos_max_retries: int = 3,
    aos_include_images: bool = False,
    aos_debug: bool = False,
    aos_strict: bool = False,
    aos_init: bool = False,
) -> Tuple[List[int], float, tools.Logbook, tools.HallOfFame]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_features = X.shape[1]
    rng_np = np.random.RandomState(seed if seed is not None else None)

    # DEAP setup
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator: 1 with probability init_prob, else 0
    toolbox.register("attr_bool", lambda: 1 if random.random() < init_prob else 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    clf = make_classifier(classifier)
    # Build fixed CV splitter if requested
    cv_for_eval: Union[int, object]
    if fixed_cv:
        base_seed = seed if seed is not None else 0
        if cv_repeats and int(cv_repeats) > 1:
            cv_for_eval = RepeatedStratifiedKFold(n_splits=cv, n_repeats=int(cv_repeats), random_state=int(base_seed))
        else:
            cv_for_eval = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(base_seed))
        try:
            print(f"[CV] Using fixed CV: {'RepeatedStratifiedKFold' if (cv_repeats and int(cv_repeats) > 1) else 'StratifiedKFold'} with seed={int(base_seed)} and n_splits={cv}")
        except Exception:
            pass
    else:
        cv_for_eval = cv
    # Default (serial) evaluation function
    def eval_wrapper(individual: List[int]):
        return evaluate_individual(individual, X, y, clf, scoring, cv_for_eval, alpha, rng_np, n_jobs_eval)
    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Parallel mapping: default to built-in map; override with multiprocessing pool if n_procs > 1
    toolbox.register("map", map)
    pool = None
    if n_procs and int(n_procs) > 1:
        try:
            import multiprocessing as _mp
            # Initialize worker context with data and settings; build classifier per worker in initializer
            worker_ctx = {
                "X": X,
                "y": y,
                "scoring": scoring,
                "cv": int(cv),
                "alpha": float(alpha),
                "n_jobs_eval": int(n_jobs_eval),
                "classifier": classifier,
                "base_seed": int(seed if seed is not None else 0),
                "fixed_cv": bool(fixed_cv),
                "cv_repeats": int(cv_repeats),
                "n_splits": int(cv),
            }
            pool = _mp.Pool(processes=int(n_procs), initializer=_init_eval_worker, initargs=(worker_ctx,))
            toolbox.unregister("map")
            toolbox.register("map", pool.map)
            # Use picklable evaluate bound to worker context
            toolbox.unregister("evaluate")
            toolbox.register("evaluate", _evaluate_individual_picklable)
            print(f"[PARALLEL] Enabled with n_procs={n_procs}; eval n_jobs={n_jobs_eval}")
        except Exception as _e:
            print(f"[PARALLEL] Failed to start pool ({_e}); falling back to serial map")

    # Dynamic operator cores (for switching): register default cores
    toolbox.register("mate_core", tools.cxTwoPoint)
    toolbox.register("mutate_core", tools.mutFlipBit, indpb=1.0 / n_features)

    # Define operator pools (function, kwargs, human-readable name)
    selection_pool = [
        (tools.selTournament, {"tournsize": 3}, "tournament"),
        (tools.selStochasticUniversalSampling, {}, "sus"),
        (tools.selBest, {}, "best"),
    ]
    # Extended crossover pool including custom operators for binary strings
    from aos.operators import default_registry as _opreg
    _reg = _opreg()
    def _cx_and_or(a, b):
        func, kwargs = _reg.bind("crossover", "and_or", {}, {"n_features": n_features})
        return func(a, b)
    def _cx_hux(a, b):
        func, kwargs = _reg.bind("crossover", "hux", {}, {"n_features": n_features})
        return func(a, b)
    def _cx_k3(a, b):
        func, kwargs = _reg.bind("crossover", "k_point", {"k": 3}, {"n_features": n_features})
        return func(a, b, **kwargs)
    crossover_pool = [
        (tools.cxOnePoint, {}, "one_point"),
        (tools.cxTwoPoint, {}, "two_point"),
        (tools.cxUniform, {"indpb": 0.5}, "uniform"),
        (_cx_k3, {}, "k_point"),
        (_cx_hux, {}, "hux"),
        (_cx_and_or, {}, "and_or"),
    ]
    def _mut_invert(ind):
        func, kwargs = _reg.bind("mutation", "invert_segment", {}, {"n_features": n_features})
        return func(ind)
    def _mut_k2(ind):
        func, kwargs = _reg.bind("mutation", "k_flip", {"k": 2}, {"n_features": n_features})
        return func(ind, **kwargs)
    mutation_pool = [
        (tools.mutFlipBit, {"indpb": 1.0 / n_features}, "flip_bit"),
        (tools.mutUniformInt, {"low": 0, "up": 1, "indpb": 1.0 / n_features}, "uniform_int"),
        (_mut_invert, {}, "invert_segment"),
        (_mut_k2, {}, "k_flip"),
    ]

    # Current operator names for logging
    current_op_sel = "tournament"
    current_op_cx = "two_point"
    current_op_mut = "flip_bit"

    # Helper to (re)bind operators on the toolbox
    def bind_select(func, kwargs, name):
        # DEAP allows re-registering to overwrite
        toolbox.register("select", func, **kwargs)

    def bind_mate_core(func, kwargs, name):
        toolbox.register("mate_core", func, **kwargs)

    def bind_mutate_core(func, kwargs, name):
        toolbox.register("mutate_core", func, **kwargs)

    # Initialize population
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ensure_valid(ind, random)

    hof = tools.HallOfFame(maxsize=1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Decorate mate/mutate to enforce validity
    def mate_and_repair(ind1, ind2):
        child1, child2 = toolbox.mate_core(ind1, ind2)
        ensure_valid(child1, random)
        ensure_valid(child2, random)
        return child1, child2

    def mutate_and_repair(ind):
        result = toolbox.mutate_core(ind)
        # DEAP mutators return (individual,), ensure we extract correctly
        mutant = result[0] if isinstance(result, tuple) else result
        ensure_valid(mutant, random)
        return (mutant,)

    toolbox.register("mate_and_repair", mate_and_repair)
    toolbox.register("mutate_and_repair", mutate_and_repair)

    # Replace default operators in the algorithm loop via custom ea
    def ea_simple_repair(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None):
        nonlocal current_op_sel, current_op_cx, current_op_mut
        logbook = tools.Logbook()
        # Include diagnostics: diversity, improvement rate, operator success rates
        logbook.header = [
            "gen",
            "nevals",
            "diversity",
            "improvement_rate",
            "stagnation",
            "sr_cx",
            "sr_mut",
            "cx_applied",
            "mut_applied",
            "op_sel",
            "op_cx",
            "op_mut",
            "op_sel_param",
            "op_cx_param",
            "op_mut_param",
            "cxpb",
            "mutpb",
            "switch_interval",
            "since_last_switch",
            "switch_effective",
        ] + (stats.fields if stats else [])
        prev_avg = None
        prev_best = None
        stagnation = 0
        # Operator params tracking (persist across gens for logging)
        current_sel_params: Dict[str, Any] = {"tournsize": 3}
        current_cx_params: Dict[str, Any] = {}
        current_mut_params: Dict[str, Any] = {"indpb": 1.0 / n_features}
        # Adaptive switching controller
        cur_interval = int(op_switch_interval)
        if adaptive_switch:
            cur_interval = max(int(switch_base_interval), 1)
        since_last_switch = 0
        cooldown_left = 0
        window_effective: List[bool] = []
        prev_diversity = None

        # One-time setup diagnostics for AOS/switching
        try:
            print(
                f"[AOS][SETUP] enable={aos_enable}, adapter_imported={AOSAdapter is not None}, "
                f"debug={aos_debug}, include_images={aos_include_images}, mode={'adaptive' if adaptive_switch else 'fixed'}, "
                f"interval={op_switch_interval}, base_interval={switch_base_interval}"
            )
            if AOSAdapter is None:
                msg = AOS_IMPORT_ERROR if 'AOS_IMPORT_ERROR' in globals() else '(unknown import error)'
                print(f"[AOS][SETUP] adapter unavailable: {msg}")
                if aos_enable and aos_strict:
                    raise SystemExit("AOS strict mode: adapter unavailable")
        except Exception:
            pass

        # Evaluate the entire population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        diversity_val = population_diversity(population)
        # No improvement rate for gen 0; no operator stats yet
        record.update({
            "diversity": diversity_val,
            "improvement_rate": float("nan"),
            "stagnation": 0,
            "sr_cx": float("nan"),
            "sr_mut": float("nan"),
            "cx_applied": 0,
            "mut_applied": 0,
            "op_sel": current_op_sel,
            "op_cx": current_op_cx,
            "op_mut": current_op_mut,
            "op_sel_param": current_sel_params,
            "op_cx_param": current_cx_params,
            "op_mut_param": current_mut_params,
            "cxpb": cxpb,
            "mutpb": mutpb,
            "switch_interval": cur_interval,
            "since_last_switch": since_last_switch,
            "switch_effective": "",
        })
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        prev_avg = record.get("avg", None)
        prev_best = record.get("max", None)
        prev_diversity = diversity_val

        # Prepare optional LLM AOS adapter and static configs
        aos_adapter = None
        aos_cfg = None
        if aos_enable and AOSAdapter is not None:
            try:
                aos_adapter = AOSAdapter(
                    endpoint=aos_endpoint,
                    api_key=aos_api_key or os.environ.get("AOS_API_KEY"),
                    model=aos_model,
                    timeout=aos_timeout,
                    max_retries=aos_max_retries,
                    include_images=aos_include_images,
                )
                from aos.config_loader import load_configs
                aos_cfg = load_configs("config")
            except Exception:
                aos_adapter = None
                aos_cfg = None

        # Optional: initial AOS decision at start (choose initial operators and rates)
        if aos_enable and aos_init and aos_adapter is not None and aos_cfg is not None:
            try:
                if aos_debug:
                    print("[AOS][INIT] Requesting initial operators and rates...")
                from aos.config_loader import build_decision_payload_from_configs
                # Include a timestamp in the initial state text so the LLM prompt varies across runs
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                init_payload = build_decision_payload_from_configs(
                    aos_cfg,
                    state_text=f"Initial stage at {ts}: please choose initial operators and cxpb/mutpb.",
                    current_iteration=0,
                )
                if aos_debug:
                    msgs_dbg = build_decision_prompt_v2(init_payload)
                    print("[AOS][REQUEST][INIT] choose_operators messages:")
                    for i, m in enumerate(msgs_dbg):
                        role = m.get("role"); content = m.get("content", "")
                        print(f"--- message {i} ({role}) ---\n{content}\n")
                data_dec0 = aos_adapter.client.chat(messages=build_decision_prompt_v2(init_payload), temperature=0.1, response_format="json_object")
                raw0 = data_dec0["choices"][0]["message"].get("content", "{}")
                if aos_debug:
                    print("[AOS][RESPONSE][INIT] raw content:\n", raw0)
                try:
                    import json as _json
                    parsed0 = _json.loads(raw0)
                except Exception:
                    s = raw0.find("{"); e = raw0.rfind("}")
                    parsed0 = _json.loads(raw0[s:e+1]) if s >= 0 and e >= 0 else {}
                decision0, warn0 = validate_decision(parsed0)
                if warn0 and aos_debug:
                    print("[AOS][POST][INIT] warnings:", warn0)
                # Bind using registry
                from aos.operators import default_registry
                reg0 = default_registry()
                ctx0 = {"n_features": n_features}
                s0 = decision0.get("Selection", {})
                c0 = decision0.get("Crossover", {})
                m0 = decision0.get("Mutation", {})
                sel0 = reg0.canonicalize("selection", s0.get("name"))
                sel_norm0 = reg0.normalize_params("selection", sel0, s0.get("parameter", {}))
                sel_func0, sel_kwargs0 = reg0.bind("selection", sel0, sel_norm0, ctx0)
                cx0 = reg0.canonicalize("crossover", c0.get("name"))
                cx_norm0 = reg0.normalize_params("crossover", cx0, c0.get("parameter", {}))
                cx_func0, cx_kwargs0 = reg0.bind("crossover", cx0, cx_norm0, ctx0)
                mut0 = reg0.canonicalize("mutation", m0.get("name"))
                mut_norm0 = reg0.normalize_params("mutation", mut0, m0.get("parameter", {}))
                mut_func0, mut_kwargs0 = reg0.bind("mutation", mut0, mut_norm0, ctx0)
                # Apply bindings
                bind_select(sel_func0, sel_kwargs0, sel0)
                bind_mate_core(cx_func0, cx_kwargs0, cx0)
                bind_mutate_core(mut_func0, mut_kwargs0, mut0)
                current_op_sel = sel0
                current_op_cx = cx0
                current_op_mut = mut0
                current_sel_params = dict(sel_kwargs0)
                current_cx_params = dict(cx_kwargs0)
                current_mut_params = dict(mut_kwargs0)
                # Rates
                cxpb = float(decision0.get("cxpb", cxpb))
                mutpb = float(decision0.get("mutpb", mutpb))
                if aos_debug:
                    print(
                        f"[AOS][APPLY][INIT] sel={current_op_sel} {current_sel_params}; "
                        f"cx={current_op_cx} {current_cx_params}; mut={current_op_mut} {current_mut_params}; "
                        f"cxpb={cxpb:.3f}, mutpb={mutpb:.3f}"
                    )
            except Exception as e:
                print("[AOS][ERROR][INIT] initial decision:", e)
                if aos_strict:
                    raise SystemExit(f"AOS strict mode: initial decision failed: {e}")

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Build a per-generation splitter for SR fairness if enabled
            sr_skf = None
            if sr_fair_cv:
                from sklearn.model_selection import StratifiedKFold as _SKF
                base_seed = 0 if seed is None else int(seed)
                sr_skf = _SKF(n_splits=cv, shuffle=True, random_state=base_seed + gen * 1009)
            # Optional operator switching at fixed intervals
            do_switch = False
            if not disable_op_switch and gen < ngen:
                if adaptive_switch:
                    if since_last_switch >= max(cur_interval, 1):
                        do_switch = True
                else:
                    if op_switch_interval and op_switch_interval > 0 and (gen % op_switch_interval == 0):
                        do_switch = True

            if do_switch:
                # Ensure latest overview plot is available before summarization/decision
                try:
                    # Always generate the default overview under output_dir
                    plot_overview(logbook, output_dir)
                except Exception:
                    pass
                # First try LLM-driven AOS if available
                sel_func = cx_func = mut_func = None
                sel_kwargs: Dict[str, Any] = {}
                cx_kwargs: Dict[str, Any] = {}
                mut_kwargs: Dict[str, Any] = {}
                sel_name = current_op_sel
                cx_name = current_op_cx
                mut_name = current_op_mut

                if aos_adapter is not None and aos_cfg is not None:
                    try:
                        from aos.config_loader import build_state_payload_from_configs, build_decision_payload_from_configs
                        # Resolve overview image path: prefer config paths.overview_image but anchor to current output_dir
                        try:
                            cfg_overview = aos_cfg.get("algo_config", {}).get("paths", {}).get("overview_image", "overview.png")
                        except Exception:
                            cfg_overview = "overview.png"
                        def _resolve_under(base_dir: str, p: str) -> str:
                            import os as _os
                            return p if _os.path.isabs(p) else _os.path.join(base_dir, p)
                        # Path where plot_overview just wrote
                        overview_default = os.path.join(output_dir, "overview.png")
                        # Target path for AOS (relative cfg path will be placed under output_dir)
                        overview_path = _resolve_under(output_dir, str(cfg_overview))
                        # If different, try to replicate/copy so that the configured path exists for AOS
                        try:
                            if overview_path != overview_default and os.path.exists(overview_default):
                                os.makedirs(os.path.dirname(overview_path), exist_ok=True)
                                import shutil as _sh
                                _sh.copy2(overview_default, overview_path)
                        except Exception:
                            pass
                        state_payload = build_state_payload_from_configs(aos_cfg, current_generation=gen, overview_image=overview_path)
                        # Inject current dynamic rates (may be changed by previous decisions)
                        try:
                            state_payload["cxpb"] = cxpb
                            state_payload["mutpb"] = mutpb
                        except Exception:
                            pass
                        print(f"[AOS] Switch at gen={gen} (interval={cur_interval}, adaptive_switching={adaptive_switch}, aos_enabled={bool(aos_adapter)})")
                        if aos_debug:
                            print(f"[AOS][SETUP] endpoint={aos_endpoint}, model={aos_model}")
                        # Summarize state: print prompt and response when debug
                        try:
                            if aos_debug:
                                state_msgs_dbg = build_state_prompt_v2(state_payload)
                                print("[AOS][REQUEST] summarize_state messages:")
                                for i, m in enumerate(state_msgs_dbg):
                                    role = m.get("role"); content = m.get("content", "")
                                    print(f"--- message {i} ({role}) ---\n{content}\n")
                                if aos_include_images:
                                    print(f"[AOS][REQUEST] summarize_state image attached: {overview_path}")
                            # Use adapter (attaches image if enabled)
                            state_text = aos_adapter.summarize_state(state_payload)
                            if aos_debug:
                                print("[AOS][RESPONSE] summarize_state raw content:\n", state_text)
                        except Exception as e:
                            print("[AOS][ERROR] summarize_state:", e)
                            if aos_strict:
                                raise SystemExit(f"AOS strict mode: summarize_state failed: {e}")
                            state_text = ""
                        # Choose operators: print prompt and response when debug
                        try:
                            dec_payload = build_decision_payload_from_configs(aos_cfg, state_text=state_text, current_iteration=gen)
                            if aos_debug:
                                decision_msgs_dbg = build_decision_prompt_v2(dec_payload)
                                print("[AOS][REQUEST] choose_operators messages:")
                                for i, m in enumerate(decision_msgs_dbg):
                                    role = m.get("role"); content = m.get("content", "")
                                    print(f"--- message {i} ({role}) ---\n{content}\n")
                            # Raw call to print response and then normalize
                            data_dec = aos_adapter.client.chat(messages=build_decision_prompt_v2(dec_payload), temperature=0.1, response_format="json_object")
                            raw_dec = data_dec["choices"][0]["message"].get("content", "{}")
                            if aos_debug:
                                print("[AOS][RESPONSE] choose_operators raw content:\n", raw_dec)
                            try:
                                import json as _json
                                parsed = _json.loads(raw_dec)
                            except Exception:
                                start = raw_dec.find("{"); end = raw_dec.rfind("}")
                                parsed = _json.loads(raw_dec[start:end+1]) if start >= 0 and end >= 0 else {}
                            decision, warn = validate_decision(parsed)
                            if warn and aos_debug:
                                print("[AOS][POST] warnings:", warn)
                            if aos_debug:
                                print("[AOS][POST] normalized decision:", decision)
                        except Exception as e:
                            print("[AOS][ERROR] choose_operators:", e)
                            if aos_strict:
                                raise SystemExit(f"AOS strict mode: choose_operators failed: {e}")
                            raise
                        # Map decision to DEAP using operator registry
                        s = decision.get("Selection", {})
                        c = decision.get("Crossover", {})
                        m = decision.get("Mutation", {})
                        try:
                            from aos.operators import default_registry
                            reg = default_registry()
                            ctx = {"n_features": n_features}
                            # Selection
                            sel_name_canon = reg.canonicalize("selection", s.get("name"))
                            sel_norm = reg.normalize_params("selection", sel_name_canon, s.get("parameter", {}))
                            sel_func, sel_kwargs = reg.bind("selection", sel_name_canon, sel_norm, ctx)
                            sel_name = sel_name_canon
                            current_sel_params = dict(sel_kwargs)
                            # Crossover
                            cx_name_canon = reg.canonicalize("crossover", c.get("name"))
                            cx_norm = reg.normalize_params("crossover", cx_name_canon, c.get("parameter", {}))
                            cx_func, cx_kwargs = reg.bind("crossover", cx_name_canon, cx_norm, ctx)
                            cx_name = cx_name_canon
                            current_cx_params = dict(cx_kwargs)
                            # Mutation
                            mut_name_canon = reg.canonicalize("mutation", m.get("name"))
                            mut_norm = reg.normalize_params("mutation", mut_name_canon, m.get("parameter", {}))
                            mut_func, mut_kwargs = reg.bind("mutation", mut_name_canon, mut_norm, ctx)
                            mut_name = mut_name_canon
                            current_mut_params = dict(mut_kwargs)
                        except Exception as e:
                            # If registry fails for any reason, leave func/kwargs as None to trigger fallback
                            if aos_strict:
                                raise SystemExit(f"AOS strict mode: registry binding failed: {e}")
                            pass
                        # Rates
                        cxpb = float(decision.get("cxpb", cxpb))
                        mutpb = float(decision.get("mutpb", mutpb))
                        if aos_debug:
                            try:
                                print(
                                    f"[AOS][APPLY] gen={gen}: "
                                    f"sel={sel_name} params={current_sel_params}; "
                                    f"cx={cx_name} params={current_cx_params}; "
                                    f"mut={mut_name} params={current_mut_params}; "
                                    f"cxpb={cxpb:.3f}, mutpb={mutpb:.3f}"
                                )
                            except Exception:
                                pass
                    except Exception:
                        if aos_strict:
                            raise
                        pass

                # Fallback to random pick different from current names
                def pick(pool, current_name):
                    candidates = [item for item in pool if item[2] != current_name]
                    base = candidates if candidates else pool
                    func, kwargs, name = random.choice(base)
                    return func, kwargs, name

                if sel_func is None:
                    if aos_enable and aos_strict:
                        raise SystemExit("AOS strict mode: selection not decided; refusing random fallback")
                    if aos_adapter is None or aos_cfg is None:
                        try:
                            print("[AOS][INFO] AOS not active; falling back to random operator pick for selection")
                        except Exception:
                            pass
                    sel_func, sel_kwargs, sel_name = pick(selection_pool, current_op_sel)
                    # Update params from picked kwargs
                    if sel_name in ("tournament", "tournament3"):
                        current_sel_params = {"tournsize": int(sel_kwargs.get("tournsize", 3))}
                    else:
                        current_sel_params = {}
                if cx_func is None:
                    if aos_enable and aos_strict:
                        raise SystemExit("AOS strict mode: crossover not decided; refusing random fallback")
                    if aos_adapter is None or aos_cfg is None:
                        try:
                            print("[AOS][INFO] AOS not active; falling back to random operator pick for crossover")
                        except Exception:
                            pass
                    cx_func, cx_kwargs, cx_name = pick(crossover_pool, current_op_cx)
                    if cx_name.startswith("uniform"):
                        current_cx_params = {"indpb": float(cx_kwargs.get("indpb", 0.5))}
                    else:
                        current_cx_params = {}
                if mut_func is None:
                    if aos_enable and aos_strict:
                        raise SystemExit("AOS strict mode: mutation not decided; refusing random fallback")
                    if aos_adapter is None or aos_cfg is None:
                        try:
                            print("[AOS][INFO] AOS not active; falling back to random operator pick for mutation")
                        except Exception:
                            pass
                    mut_func, mut_kwargs, mut_name = pick(mutation_pool, current_op_mut)
                    if mut_name == "flip_bit":
                        current_mut_params = {"indpb": float(mut_kwargs.get("indpb", 1.0 / n_features))}
                    elif mut_name == "uniform_int":
                        current_mut_params = {
                            "low": int(mut_kwargs.get("low", 0)),
                            "up": int(mut_kwargs.get("up", 1)),
                            "indpb": float(mut_kwargs.get("indpb", 1.0 / n_features)),
                        }
                    else:
                        current_mut_params = {}

                bind_select(sel_func, sel_kwargs, sel_name)
                bind_mate_core(cx_func, cx_kwargs, cx_name)
                bind_mutate_core(mut_func, mut_kwargs, mut_name)

                # Update current names
                current_op_sel = sel_name
                current_op_cx = cx_name
                current_op_mut = mut_name
                # Reset controller counters
                if adaptive_switch:
                    since_last_switch = 0
                    cooldown_left = max(int(switch_cooldown), 0)
                    window_effective.clear()

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation with operator success tracking
            cx_applied = 0
            cx_success = 0
            mut_applied = 0
            mut_success = 0
            # Use index-based tracking to be robust to object identity changes
            cx_baseline_idx = {}
            mut_baseline_idx = {}
            crossed_idx = set()

            # Crossover: compare children to max(parent fitness)
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    p1 = offspring[i - 1]
                    p2 = offspring[i]
                    # Parents are clones from selected population; should have valid fitness
                    if sr_skf is not None:
                        # Evaluate parents under generation-fixed CV for fair SR
                        def _fair_eval_parent(ind):
                            mask = np.array(ind, dtype=bool)
                            if not mask.any():
                                return -1e6
                            X_sel = X[:, mask]
                            try:
                                scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=sr_skf, n_jobs=n_jobs_eval)
                                mean_score = float(np.mean(scores))
                            except Exception:
                                return -1e6
                            penalty = alpha * (mask.sum() / mask.size)
                            return mean_score - penalty
                        parent_best = max(_fair_eval_parent(p1), _fair_eval_parent(p2))
                    else:
                        parent_best = max(
                            p1.fitness.values[0] if p1.fitness.valid else -1e12,
                            p2.fitness.values[0] if p2.fitness.valid else -1e12,
                        )
                    offspring[i - 1], offspring[i] = toolbox.mate_and_repair(p1, p2)
                    # Track baselines by index
                    cx_baseline_idx[i - 1] = parent_best
                    cx_baseline_idx[i] = parent_best
                    crossed_idx.add(i - 1)
                    crossed_idx.add(i)
                    cx_applied += 2
                    # Invalidate fitness
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            # Mutation: compare mutated child to its immediate parent fitness.
            # To avoid double-credit ambiguity, we only count mutation events on individuals
            # that did not undergo crossover in this generation.
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    ind = offspring[i]
                    if (i not in crossed_idx) and ind.fitness.valid:
                        if sr_skf is not None:
                            mask = np.array(ind, dtype=bool)
                            if not mask.any():
                                baseline_val = -1e6
                            else:
                                X_sel = X[:, mask]
                                try:
                                    scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=sr_skf, n_jobs=n_jobs_eval)
                                    mean_score = float(np.mean(scores))
                                except Exception:
                                    mean_score = -1e6
                                baseline_val = mean_score - alpha * (mask.sum() / mask.size)
                        else:
                            baseline_val = ind.fitness.values[0]
                        mut_baseline_idx[i] = baseline_val
                        mut_applied += 1
                    (offspring[i],) = toolbox.mutate_and_repair(ind)
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Compute operator success rates after evaluation (index-based)
            if cx_applied > 0:
                for idx, base in cx_baseline_idx.items():
                    ind = offspring[idx]
                    if sr_skf is not None:
                        mask = np.array(ind, dtype=bool)
                        if not mask.any():
                            new_fit = -1e6
                        else:
                            X_sel = X[:, mask]
                            try:
                                scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=sr_skf, n_jobs=n_jobs_eval)
                                mean_score = float(np.mean(scores))
                            except Exception:
                                mean_score = -1e6
                            new_fit = mean_score - alpha * (mask.sum() / mask.size)
                    else:
                        new_fit = ind.fitness.values[0]
                    if new_fit > base:
                        cx_success += 1
                sr_cx = cx_success / cx_applied if cx_applied > 0 else float("nan")
            else:
                sr_cx = float("nan")
            if mut_applied > 0:
                for idx, base in mut_baseline_idx.items():
                    ind = offspring[idx]
                    if sr_skf is not None:
                        mask = np.array(ind, dtype=bool)
                        if not mask.any():
                            new_fit = -1e6
                        else:
                            X_sel = X[:, mask]
                            try:
                                scores = cross_val_score(clf, X_sel, y, scoring=scoring, cv=sr_skf, n_jobs=n_jobs_eval)
                                mean_score = float(np.mean(scores))
                            except Exception:
                                mean_score = -1e6
                            new_fit = mean_score - alpha * (mask.sum() / mask.size)
                    else:
                        new_fit = ind.fitness.values[0]
                    if new_fit > base:
                        mut_success += 1
                sr_mut = mut_success / mut_applied if mut_applied > 0 else float("nan")
            else:
                sr_mut = float("nan")

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            diversity_val = population_diversity(population)
            avg_now = record.get("avg", None)
            best_now = record.get("max", None)
            if prev_avg is None or avg_now is None or abs(prev_avg) < 1e-12:
                ir = float("nan")
            else:
                ir = float((avg_now - prev_avg) / prev_avg)
            # Stagnation: consecutive gens with unchanged best fitness
            if prev_best is None or best_now is None:
                stagnation = 0
            else:
                try:
                    import math
                    if math.isfinite(prev_best) and math.isfinite(best_now) and abs(best_now - prev_best) <= 1e-12:
                        stagnation += 1
                    else:
                        stagnation = 0
                except Exception:
                    stagnation = 0
            # Adaptive interval update
            effective_flag = ""
            if adaptive_switch:
                # simple effectiveness using IR/SR/D signals
                ir_ok = (isinstance(ir, float) and ir == ir) and (ir > switch_ir_thresh)
                sr_vals = []
                if sr_cx == sr_cx:
                    sr_vals.append(sr_cx)
                if sr_mut == sr_mut:
                    sr_vals.append(sr_mut)
                sr_mean = (sum(sr_vals) / len(sr_vals)) if sr_vals else 0.0
                sr_ok = sr_mean > switch_sr_thresh
                delta_d = (diversity_val - prev_diversity) if (prev_diversity == prev_diversity) else 0.0
                d_ok = delta_d >= switch_deltaD_thresh
                current_ok = (ir_ok or sr_ok or d_ok)
                if cooldown_left > 0:
                    cooldown_left -= 1
                else:
                    window_effective.append(bool(current_ok))
                    if len(window_effective) > max(int(switch_window), 1):
                        window_effective = window_effective[-int(switch_window):]
                    if len(window_effective) == max(int(switch_window), 1):
                        ineffective = window_effective.count(False)
                        effective = window_effective.count(True)
                        if ineffective >= max(int(switch_patience), 1):
                            cur_interval = min(int(cur_interval * 2), int(switch_max_interval))
                            window_effective.clear()
                            effective_flag = "backoff"
                        elif effective >= 1 and cur_interval > int(switch_base_interval):
                            cur_interval = max(int(switch_base_interval), int(max(1, cur_interval // 2)))
                            window_effective.clear()
                            effective_flag = "recover"
                if not effective_flag:
                    effective_flag = "ok" if current_ok else "no"
                since_last_switch += 1

            # Serialize operator params as JSON strings for CSV readability
            try:
                import json as _json
                sel_param_str = _json.dumps(current_sel_params)
                cx_param_str = _json.dumps(current_cx_params)
                mut_param_str = _json.dumps(current_mut_params)
            except Exception:
                sel_param_str = str(current_sel_params)
                cx_param_str = str(current_cx_params)
                mut_param_str = str(current_mut_params)

            record.update({
                "diversity": diversity_val,
                "improvement_rate": ir,
                "stagnation": stagnation,
                "sr_cx": sr_cx,
                "sr_mut": sr_mut,
                "cx_applied": cx_applied,
                "mut_applied": mut_applied,
                "op_sel": current_op_sel,
                "op_cx": current_op_cx,
                "op_mut": current_op_mut,
                "op_sel_param": sel_param_str,
                "op_cx_param": cx_param_str,
                "op_mut_param": mut_param_str,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "switch_interval": cur_interval if not disable_op_switch else "",
                "since_last_switch": since_last_switch if not disable_op_switch else "",
                "switch_effective": effective_flag,
            })
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            prev_avg = avg_now
            prev_best = best_now
            prev_diversity = diversity_val

        return population, logbook

    # Run GA
    try:
        _, logbook = ea_simple_repair(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof)

        best_ind = hof[0]
        best_mask = list(map(int, best_ind))
        best_fit = float(best_ind.fitness.values[0])
        return best_mask, best_fit, logbook, hof
    finally:
        # Clean up worker pool if created
        try:
            if pool is not None:
                pool.close()
                pool.join()
        except Exception:
            pass


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA-based feature selection using DEAP")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--csv", type=str, help="Path to CSV file")
    src.add_argument("--sklearn-dataset", type=str, help="Built-in dataset: breast_cancer|iris|wine")
    src.add_argument("--openml-name", type=str, help="OpenML dataset by name (requires network)")
    src.add_argument("--openml-id", type=str, help="OpenML dataset by numeric ID (requires network)")
    p.add_argument("--target-col", type=str, help="Target column name (with --csv)")
    p.add_argument("--openml-version", type=int, default=None, help="Specific OpenML dataset version")
    p.add_argument("--data-home", type=str, default=None, help="Cache directory for OpenML downloads")

    p.add_argument("--classifier", type=str, default="logistic", choices=["logistic", "svm", "rf"], help="Classifier to evaluate subsets")
    p.add_argument("--scoring", type=str, default="accuracy", help="sklearn scoring metric (e.g., accuracy, f1_macro)")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--alpha", type=float, default=0.02, help="Sparsity penalty strength (higher => fewer features)")

    p.add_argument("--pop-size", type=int, default=60, help="Population size")
    p.add_argument("--generations", type=int, default=40, help="Number of generations")
    p.add_argument("--cxpb", type=float, default=0.6, help="Crossover probability")
    p.add_argument("--mutpb", type=float, default=0.2, help="Mutation probability")
    p.add_argument("--init-prob", type=float, default=0.1, help="Initial probability a feature is selected")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    # CV control
    p.add_argument("--fixed-cv", action="store_true", help="Use a fixed CV splitter (seeded) during GA evaluation for determinism")
    p.add_argument("--cv-repeats", type=int, default=1, help="If >1 with --fixed-cv, use RepeatedStratifiedKFold with this many repeats")
    p.add_argument("--n-procs", type=int, default=1, help="Worker processes for parallel evaluation (DEAP map)")
    p.add_argument("--eval-n-jobs", type=int, default=1, help="n_jobs for sklearn cross_val_score inside each evaluation")

    p.add_argument("--output", type=str, default="ga_results", help="Directory to save outputs")
    p.add_argument("--use-config", action="store_true", help="Load run settings from config/*.json instead of CLI flags")
    p.add_argument("--config-dir", type=str, default="config", help="Config directory (task/operator/algo)")
    # Operator switching
    p.add_argument("--op-switch-interval", type=int, default=10, help="Switch selection/crossover/mutation operators every K generations (0 to disable)")
    p.add_argument("--disable-op-switch", action="store_true", help="Disable operator switching even if interval > 0")
    p.add_argument("--sr-fair-cv", action="store_true", help="Use generation-fixed CV splitter for SR fairness comparisons")
    # Adaptive switch controller (pluggable)
    p.add_argument("--adaptive-switch", action="store_true", help="Enable adaptive (backoff) control of operator switch interval")
    p.add_argument("--switch-base-interval", type=int, default=10, help="Base interval K0 for adaptive switching")
    p.add_argument("--switch-min-interval", type=int, default=5, help="Minimum adaptive switch interval")
    p.add_argument("--switch-max-interval", type=int, default=40, help="Maximum adaptive switch interval")
    p.add_argument("--switch-window", type=int, default=5, help="Window size (generations) to judge switch effectiveness")
    p.add_argument("--switch-patience", type=int, default=3, help="Backoff if ineffective generations in window >= patience")
    p.add_argument("--switch-ir-thresh", type=float, default=0.0005, help="IR threshold to consider improvement positive")
    p.add_argument("--switch-sr-thresh", type=float, default=0.05, help="Average SR threshold ((sr_cx+sr_mut)/2)")
    p.add_argument("--switch-deltaD-thresh", type=float, default=0.0, help="Diversity delta threshold to consider non-decreasing beneficial")
    p.add_argument("--switch-cooldown", type=int, default=1, help="Generations to ignore effectiveness right after a switch")
    # Visualization smoothing
    p.add_argument("--ema-alpha-ir", type=float, default=0.3, help="EMA smoothing alpha for IR plots")
    p.add_argument("--ema-alpha-sr", type=float, default=0.25, help="EMA smoothing alpha for SR plots")
    p.add_argument("--ema-alpha-counts", type=float, default=0.3, help="EMA smoothing alpha for operator counts plots")
    # LLM-driven adaptive operator selection (pluggable)
    p.add_argument("--aos-enable", action="store_true", help="Enable LLM-driven operator selection at switch points")
    p.add_argument("--aos-endpoint", type=str, default=os.environ.get("AOS_ENDPOINT", "https://api.sunxie.xyz"))
    p.add_argument("--aos-model", type=str, default=os.environ.get("AOS_MODEL", "gpt-4o-mini"))
    p.add_argument("--aos-api-key-env", type=str, default="AOS_API_KEY", help="Env var name holding API key")
    p.add_argument("--aos-timeout", type=float, default=30.0)
    p.add_argument("--aos-max-retries", type=int, default=3)
    p.add_argument("--aos-include-images", action="store_true", help="Include charts as base64 in prompts")
    p.add_argument("--aos-debug", action="store_true", help="Print AOS prompts and responses for debugging")
    p.add_argument("--aos-strict", action="store_true", help="Fail fast at switch points if AOS is unavailable or errors")
    p.add_argument("--aos-init", action="store_true", help="At start, ask LLM to choose initial operators and cxpb/mutpb")
    p.add_argument("--aos-config-dir", type=str, default="config", help="Directory holding task/operator/algo config JSONs")
    return p.parse_args()


def save_results(
    output_dir: str,
    best_mask: List[int],
    best_fit: float,
    logbook: tools.Logbook,
    feature_names: List[str],
):
    os.makedirs(output_dir, exist_ok=True)

    selected_idx = [i for i, b in enumerate(best_mask) if b == 1]
    selected_names = [feature_names[i] for i in selected_idx]

    with open(os.path.join(output_dir, "best_solution.json"), "w") as f:
        json.dump({
            "best_fitness": best_fit,
            "selected_indices": selected_idx,
            "selected_features": selected_names,
            "num_selected": len(selected_idx),
            "total_features": len(feature_names),
        }, f, indent=2)

    # Save logbook as CSV
    log_df = pd.DataFrame(logbook)
    log_df.to_csv(os.path.join(output_dir, "evolution_log.csv"), index=False)


def plot_best_fitness(logbook: tools.Logbook, output_dir: str) -> None:
    """Generate a line plot of best fitness (max) over generations.
    Saves to <output_dir>/best_fitness.png. If matplotlib is unavailable, skip gracefully.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # Matplotlib not installed or not usable in this environment; skip plotting.
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logbook)
    # Detect operator switch points where any operator name changes
    switch_gens: List[int] = []
    if {"gen", "op_sel", "op_cx", "op_mut"}.issubset(df.columns):
        prev = None
        for _, row in df.iterrows():
            cur = (row["op_sel"], row["op_cx"], row["op_mut"])
            g = int(row["gen"]) if "gen" in row else None
            if prev is not None and cur != prev and g is not None:
                switch_gens.append(g)
            prev = cur
    if not {"gen", "max"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["gen"], df["max"], marker="o", linewidth=1.8)
    # Vertical lines at switch points
    for g in switch_gens:
        ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (max)")
    ax.set_title("GA Best Fitness over Generations")
    ax.grid(True, alpha=0.3)
    # Avoid tight_layout warnings
    out_path = os.path.join(output_dir, "best_fitness.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_diversity(logbook: tools.Logbook, output_dir: str) -> None:
    """Generate a line plot of population diversity over generations with a
    horizontal line at the historical mean (ignoring NaNs).
    Saves to <output_dir>/diversity.png. If matplotlib is unavailable, skip.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logbook)
    switch_gens: List[int] = []
    if {"gen", "op_sel", "op_cx", "op_mut"}.issubset(df.columns):
        prev = None
        for _, row in df.iterrows():
            cur = (row["op_sel"], row["op_cx"], row["op_mut"])
            g = int(row["gen"]) if "gen" in row else None
            if prev is not None and cur != prev and g is not None:
                switch_gens.append(g)
            prev = cur
    if not {"gen", "diversity"}.issubset(df.columns):
        return
    y = df["diversity"].astype(float)
    mean_y = float(np.nanmean(y.values)) if len(y) else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["gen"], y, marker="o", linewidth=1.8, label="diversity")
    for g in switch_gens:
        ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    if mean_y == mean_y:  # not NaN
        ax.axhline(mean_y, color="red", linestyle="--", alpha=0.7, label=f"mean={mean_y:.3f}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity (avg Hamming fraction)")
    ax.set_title("GA Population Diversity over Generations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Avoid tight_layout warnings
    out_path = os.path.join(output_dir, "diversity.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_improvement_rate(logbook: tools.Logbook, output_dir: str, ema_alpha: float = 0.3) -> None:
    """Generate a bar/line plot of improvement rate (IR) over generations.
    Positive IR values colored green, negative in red, NaN/zero in gray.
    Saves to <output_dir>/improvement_rate.png. If matplotlib unavailable, skip.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logbook)
    switch_gens: List[int] = []
    if {"gen", "op_sel", "op_cx", "op_mut"}.issubset(df.columns):
        prev = None
        for _, row in df.iterrows():
            cur = (row["op_sel"], row["op_cx"], row["op_mut"])
            g = int(row["gen"]) if "gen" in row else None
            if prev is not None and cur != prev and g is not None:
                switch_gens.append(g)
            prev = cur
    if not {"gen", "improvement_rate"}.issubset(df.columns):
        return
    x = df["gen"].values
    y = df["improvement_rate"].astype(float).values
    # Colors by sign
    colors = []
    for v in y:
        if not np.isfinite(v) or abs(v) < 1e-12:
            colors.append("#7f7f7f")  # gray for NaN or ~0
        elif v > 0:
            colors.append("#2ca02c")  # green
        else:
            colors.append("#d62728")  # red
    def ema(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        out = np.empty_like(series, dtype=float)
        m = np.nan
        for i, v in enumerate(series):
            v = 0.0 if not np.isfinite(v) else v
            m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
            out[i] = m
        return out

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, np.nan_to_num(y, nan=0.0), color=colors, alpha=0.7, label="IR")
    # Overlay EMA-smoothed IR for trend
    y_ema = ema(y, alpha=float(ema_alpha))
    ax.plot(x, y_ema, color="#1f77b4", linewidth=2.0, alpha=0.9, label="EMA(0.3)")
    for g in switch_gens:
        ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Improvement Rate (IR)")
    ax.set_title("GA Fitness Improvement Rate per Generation")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.grid(True, axis="y", alpha=0.3)
    # Optional zoom inset for last K generations
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        K = min(10, len(x))
        if K >= 3:
            ax_ins = inset_axes(ax, width="38%", height="45%", loc="upper right", borderpad=1.0)
            xz = x[-K:]
            yz = y[-K:]
            colors_z = []
            for v in yz:
                if not np.isfinite(v) or abs(v) < 1e-12:
                    colors_z.append("#7f7f7f")
                elif v > 0:
                    colors_z.append("#2ca02c")
                else:
                    colors_z.append("#d62728")
            ax_ins.bar(xz, np.nan_to_num(yz, nan=0.0), color=colors_z, alpha=0.7)
            ax_ins.plot(xz, ema(yz, alpha=0.3), color="#1f77b4", linewidth=1.6)
            # Mark switches in inset if within window
            for g in switch_gens:
                if g >= xz[0]:
                    ax_ins.axvline(g, color="#666666", linestyle=":", linewidth=0.8, alpha=0.8)
            ax_ins.set_title(f"Last {K}", fontsize=9)
            ax_ins.axhline(0.0, color="black", linewidth=0.8)
            ax_ins.tick_params(axis='both', labelsize=8)
            ax_ins.grid(True, axis="y", alpha=0.25)
    except Exception:
        pass

    # Place legend outside to avoid covering bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, framealpha=0.9)
    # Leave room at bottom for legend (avoid tight_layout warnings)
    fig.subplots_adjust(bottom=0.22)
    out_path = os.path.join(output_dir, "improvement_rate.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_operator_success(logbook: tools.Logbook, output_dir: str, ema_alpha_sr: float = 0.25, ema_alpha_counts: float = 0.3) -> None:
    """Plot operator success with two aligned panels for clarity on long runs:
    - Top: sr_cx/sr_mut lines with EMA overlays.
    - Bottom: cx_applied/mut_applied bars (counts).
    Saves to operator_success.png. If matplotlib unavailable, skip.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logbook)
    required = {"gen", "sr_cx", "sr_mut", "cx_applied", "mut_applied"}
    if not required.issubset(df.columns):
        return
    x = df["gen"].values
    sr_cx = df["sr_cx"].astype(float).values
    sr_mut = df["sr_mut"].astype(float).values
    cx_cnt = df["cx_applied"].astype(float).values
    mut_cnt = df["mut_applied"].astype(float).values
    # Switch gens
    switch_gens: List[int] = []
    if {"op_sel", "op_cx", "op_mut"}.issubset(df.columns):
        prev = None
        for _, row in df.iterrows():
            cur = (row["op_sel"], row["op_cx"], row["op_mut"])
            g = int(row["gen"]) if "gen" in row else None
            if prev is not None and cur != prev and g is not None:
                switch_gens.append(g)
            prev = cur

    def ema(series: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        out = np.empty_like(series, dtype=float)
        m = np.nan
        for i, v in enumerate(series):
            v = np.nan if not np.isfinite(v) else v
            m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
            out[i] = m
        # Fill NaNs with zeros for plotting
        out = np.nan_to_num(out, nan=0.0)
        return out

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9.5, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # Top panel: SR lines with EMA
    ax1 = fig.add_subplot(gs[0, 0])
    line1_raw, = ax1.plot(x, sr_cx, color="#9ecae1", linewidth=1.0, alpha=0.8, label="sr_cx (raw)")
    line2_raw, = ax1.plot(x, sr_mut, color="#fdd0a2", linewidth=1.0, alpha=0.8, label="sr_mut (raw)")
    line1, = ax1.plot(x, ema(sr_cx, alpha=float(ema_alpha_sr)), marker="o", linewidth=2.2, color="#1f77b4", label=f"sr_cx EMA({ema_alpha_sr})")
    line2, = ax1.plot(x, ema(sr_mut, alpha=float(ema_alpha_sr)), marker="s", linewidth=2.2, color="#ff7f0e", label=f"sr_mut EMA({ema_alpha_sr})")
    ax1.set_ylabel("Success Rate (SR)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", ncol=2, fontsize=9)
    ax1.set_title("Operator Success Rates (EMA overlay)")
    for g in switch_gens:
        ax1.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)

    # Bottom panel: counts bars
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    width = 0.4
    bars1 = ax2.bar(x - width/2, cx_cnt, width=width, color="#aec7e8", alpha=0.7, edgecolor="#6b8fb4", linewidth=0.8, label="cx_applied")
    bars2 = ax2.bar(x + width/2, mut_cnt, width=width, color="#ffbb78", alpha=0.7, edgecolor="#c27a30", linewidth=0.8, label="mut_applied")
    # Add trend lines (EMA) over counts
    def ema_counts(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        out = np.empty_like(series, dtype=float)
        m = np.nan
        for i, v in enumerate(series):
            v = 0.0 if not np.isfinite(v) else v
            m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
            out[i] = m
        return out
    ax2.plot(x, ema_counts(cx_cnt, alpha=float(ema_alpha_counts)), color="#1f77b4", linewidth=2.0, alpha=0.9, label=f"cx trend EMA({ema_alpha_counts})")
    ax2.plot(x, ema_counts(mut_cnt, alpha=float(ema_alpha_counts)), color="#ff7f0e", linewidth=2.0, alpha=0.9, label=f"mut trend EMA({ema_alpha_counts})")
    ax2.set_ylabel("Applications")
    ax2.set_xlabel("Generation")
    ax2.grid(True, axis="y", alpha=0.3)
    for g in switch_gens:
        ax2.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    # Place legend outside below the bars to avoid overlap
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8, framealpha=0.9)

    # Avoid tight_layout to prevent warnings with complex grid
    out_path = os.path.join(output_dir, "operator_success.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overview(logbook: tools.Logbook, output_dir: str, ema_alpha_ir: float = 0.3, ema_alpha_sr: float = 0.25, ema_alpha_counts: float = 0.3) -> None:
    """Create a 2x2 overview figure combining best fitness, diversity,
    improvement rate, and operator success + counts.
    Saves to <output_dir>/overview.png. If matplotlib unavailable, skip.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(logbook)
    if "gen" not in df.columns:
        return

    x = df["gen"].values
    # Detect operator switch points (gens where any operator changes)
    switch_gens: List[int] = []
    if {"op_sel", "op_cx", "op_mut"}.issubset(df.columns):
        prev = None
        for _, row in df.iterrows():
            cur = (row["op_sel"], row["op_cx"], row["op_mut"])
            g = int(row["gen"]) if "gen" in row else None
            if prev is not None and cur != prev and g is not None:
                switch_gens.append(g)
            prev = cur

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    # 1) Best fitness
    if {"max"}.issubset(df.columns):
        ax = axs[0, 0]
        ax.plot(x, df["max"].values, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_title("Best Fitness (max)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.grid(True, alpha=0.3)
        for g in switch_gens:
            ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    else:
        axs[0, 0].set_visible(False)

    # 2) Diversity with mean line
    if {"diversity"}.issubset(df.columns):
        ax = axs[0, 1]
        y = df["diversity"].astype(float).values
        mean_y = float(np.nanmean(y)) if len(y) else float("nan")
        ax.plot(x, y, marker="o", linewidth=2.0, color="#2ca02c", label="diversity")
        if np.isfinite(mean_y):
            ax.axhline(mean_y, color="red", linestyle="--", alpha=0.7, label=f"mean={mean_y:.3f}")
        ax.set_title("Population Diversity")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Avg Hamming (fraction)")
        ax.grid(True, alpha=0.3)
        for g in switch_gens:
            ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.legend(fontsize=9)
    else:
        axs[0, 1].set_visible(False)

    # 3) Improvement rate (bar + EMA + inset), consistent with standalone
    if {"improvement_rate"}.issubset(df.columns):
        ax = axs[1, 0]
        y = df["improvement_rate"].astype(float).values
        colors = []
        for v in y:
            if not np.isfinite(v) or abs(v) < 1e-12:
                colors.append("#7f7f7f")
            elif v > 0:
                colors.append("#2ca02c")
            else:
                colors.append("#d62728")
        ax.bar(x, np.nan_to_num(y, nan=0.0), color=colors, alpha=0.7, label="IR")
        # EMA trend for IR
        def ema(series: np.ndarray, alpha: float = 0.3) -> np.ndarray:
            out = np.empty_like(series, dtype=float)
            m = np.nan
            for i, v in enumerate(series):
                v = 0.0 if not np.isfinite(v) else v
                m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
                out[i] = m
            return out
        ax.plot(x, ema(y, alpha=float(ema_alpha_ir)), color="#1f77b4", linewidth=1.8, alpha=0.9, label=f"EMA({ema_alpha_ir})")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title("Improvement Rate (IR)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("IR")
        ax.grid(True, axis="y", alpha=0.3)
        for g in switch_gens:
            ax.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        # Inset for last K generations
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            K = min(10, len(x))
            if K >= 3:
                ax_ins = inset_axes(ax, width="38%", height="45%", loc="upper right", borderpad=1.0)
                xz = x[-K:]
                yz = y[-K:]
                colors_z = []
                for v in yz:
                    if not np.isfinite(v) or abs(v) < 1e-12:
                        colors_z.append("#7f7f7f")
                    elif v > 0:
                        colors_z.append("#2ca02c")
                    else:
                        colors_z.append("#d62728")
                ax_ins.bar(xz, np.nan_to_num(yz, nan=0.0), color=colors_z, alpha=0.7)
                ax_ins.plot(xz, ema(yz, alpha=float(ema_alpha_ir)), color="#1f77b4", linewidth=1.4)
                ax_ins.set_title(f"Last {K}", fontsize=9)
                ax_ins.axhline(0.0, color="black", linewidth=0.8)
                ax_ins.tick_params(axis='both', labelsize=8)
                ax_ins.grid(True, axis="y", alpha=0.25)
        except Exception:
            pass
        # External legend to avoid covering bars
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, framealpha=0.9)
    else:
        axs[1, 0].set_visible(False)

    # 4) Operator success + counts: split into stacked panels inside the tile
    if {"sr_cx", "sr_mut", "cx_applied", "mut_applied"}.issubset(df.columns):
        host_ax = axs[1, 1]
        sr_cx = df["sr_cx"].astype(float).values
        sr_mut = df["sr_mut"].astype(float).values
        cx_cnt = df["cx_applied"].astype(float).values
        mut_cnt = df["mut_applied"].astype(float).values
        pos = host_ax.get_position(fig)
        host_ax.set_visible(False)
        # Create stacked axes for SR (top) and Applications (bottom)
        ax_top = fig.add_axes([pos.x0, pos.y0 + pos.height * 0.47, pos.width, pos.height * 0.53])
        ax_bot = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * 0.42], sharex=ax_top)

        # Helper EMA
        def ema(series: np.ndarray, alpha: float) -> np.ndarray:
            out = np.empty_like(series, dtype=float)
            m = np.nan
            for i, v in enumerate(series):
                v = np.nan if not np.isfinite(v) else v
                m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
                out[i] = m
            return np.nan_to_num(out, nan=0.0)

        # Top: SR raw + EMA
        ax_top.plot(x, sr_cx, color="#9ecae1", linewidth=1.0, alpha=0.8, label="sr_cx (raw)")
        ax_top.plot(x, sr_mut, color="#fdd0a2", linewidth=1.0, alpha=0.8, label="sr_mut (raw)")
        ax_top.plot(x, ema(sr_cx, alpha=float(ema_alpha_sr)), marker="o", linewidth=2.0, color="#1f77b4", label=f"sr_cx EMA({ema_alpha_sr})")
        ax_top.plot(x, ema(sr_mut, alpha=float(ema_alpha_sr)), marker="s", linewidth=2.0, color="#ff7f0e", label=f"sr_mut EMA({ema_alpha_sr})")
        ax_top.set_ylabel("SR")
        # Hide top panel x tick labels to avoid overlap with bottom panel
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left", fontsize=8, ncol=2)
        ax_top.set_title("Operator Success Rates (raw + EMA)")
        for g in switch_gens:
            ax_top.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)

        # Bottom: counts bars + EMA trend lines
        width = 0.34
        ax_bot.bar(x - width/2, cx_cnt, width=width, color="#aec7e8", alpha=0.6, edgecolor="#6b8fb4", linewidth=0.6, label="cx_applied")
        ax_bot.bar(x + width/2, mut_cnt, width=width, color="#ffbb78", alpha=0.6, edgecolor="#c27a30", linewidth=0.6, label="mut_applied")
        def ema_counts(series: np.ndarray, alpha: float) -> np.ndarray:
            out = np.empty_like(series, dtype=float)
            m = np.nan
            for i, v in enumerate(series):
                v = 0.0 if not np.isfinite(v) else v
                m = v if not np.isfinite(m) else (alpha * v + (1 - alpha) * m)
                out[i] = m
            return out
        ax_bot.plot(x, ema_counts(cx_cnt, alpha=float(ema_alpha_counts)), color="#1f77b4", linewidth=1.6, alpha=0.9, label=f"cx trend EMA({ema_alpha_counts})")
        ax_bot.plot(x, ema_counts(mut_cnt, alpha=float(ema_alpha_counts)), color="#ff7f0e", linewidth=1.6, alpha=0.9, label=f"mut trend EMA({ema_alpha_counts})")
        ax_bot.set_ylabel("Applications")
        ax_bot.set_xlabel("Generation")
        # Ensure bottom panel doesn't draw top x labels
        ax_bot.tick_params(axis='x', which='both', top=False)
        ax_bot.grid(True, axis="y", alpha=0.3)
        for g in switch_gens:
            ax_bot.axvline(g, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
        ax_bot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=7, framealpha=0.9)
    else:
        axs[1, 1].set_visible(False)

    # Manual layout adjustments; avoid tight_layout to prevent warnings with add_axes
    fig.subplots_adjust(bottom=0.12, right=0.88)
    out_path = os.path.join(output_dir, "overview.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    # If using config, override CLI with JSON configs
    cfgs = None
    if args.use_config:
        try:
            from aos.config_loader import load_configs
            cfgs = load_configs(args.config_dir)
        except Exception:
            cfgs = None

    # Resolve dataset source
    if cfgs is not None:
        ti = cfgs.get("task_info", {})
        ds_name = ti.get("dataset_name") or ""
        ds_source = (ti.get("dataset_source") or "").lower()
        csv_info = ti.get("csv", {})
        openml_info = ti.get("openml", {})
        if ds_source.startswith("sklearn") or ds_name.startswith("sklearn:"):
            ds = ds_name.split(":", 1)[1] if ":" in ds_name else ds_name
            data = load_dataset(
                csv=None,
                target_col=None,
                sklearn_dataset=ds,
                openml_name=None,
                openml_id=None,
                openml_version=None,
                data_home=None,
            )
            try:
                print(f"[DATA] Loaded sklearn dataset '{ds}' with X.shape={data.X.shape}, y.shape={data.y.shape}")
            except Exception:
                pass
        elif ds_source == "openml":
            oname = openml_info.get("name")
            oid = openml_info.get("id")
            data = load_dataset(
                csv=None,
                target_col=None,
                sklearn_dataset=None,
                openml_name=oname,
                openml_id=str(oid) if oid is not None else None,
                openml_version=openml_info.get("version"),
                data_home=args.data_home,
            )
            try:
                src = f"name='{oname}'" if oname else f"id={oid}"
                print(f"[DATA] Loaded OpenML dataset ({src}) with X.shape={data.X.shape}, y.shape={data.y.shape}")
                if args.data_home:
                    print(f"[DATA] OpenML cache dir: {args.data_home}")
            except Exception:
                pass
        elif ds_source == "csv":
            data = load_dataset(
                csv=csv_info.get("path"),
                target_col=csv_info.get("target_col"),
                sklearn_dataset=None,
                openml_name=None,
                openml_id=None,
                openml_version=None,
                data_home=None,
            )
            try:
                print(f"[DATA] Loaded CSV dataset from '{csv_info.get('path')}' with X.shape={data.X.shape}, y.shape={data.y.shape}")
            except Exception:
                pass
        else:
            raise ValueError("Config dataset_source invalid or unsupported")
    else:
        if not any([args.csv, args.sklearn_dataset, args.openml_name, args.openml_id]):
            raise SystemExit("Provide dataset via CLI or use --use-config")
        data = load_dataset(
            csv=args.csv,
            target_col=args.target_col,
            sklearn_dataset=args.sklearn_dataset,
            openml_name=args.openml_name,
            openml_id=args.openml_id,
            openml_version=args.openml_version,
            data_home=args.data_home,
        )
        try:
            if args.openml_id or args.openml_name:
                src = f"id={args.openml_id}" if args.openml_id else f"name='{args.openml_name}'"
                print(f"[DATA] Loaded OpenML dataset ({src}) with X.shape={data.X.shape}, y.shape={data.y.shape}")
                if args.data_home:
                    print(f"[DATA] OpenML cache dir: {args.data_home}")
            elif args.sklearn_dataset:
                print(f"[DATA] Loaded sklearn dataset '{args.sklearn_dataset}' with X.shape={data.X.shape}, y.shape={data.y.shape}")
            elif args.csv:
                print(f"[DATA] Loaded CSV dataset from '{args.csv}' with X.shape={data.X.shape}, y.shape={data.y.shape}")
        except Exception:
            pass

    # Resolve runtime settings (either from config or CLI)
    aos_api_key = os.environ.get(args.aos_api_key_env, "")
    pop_size = args.pop_size
    generations = args.generations
    cv = args.cv
    alpha = args.alpha
    scoring = args.scoring
    classifier = args.classifier
    cxpb = args.cxpb
    mutpb = args.mutpb
    # Parallelism defaults; may be overridden by config later
    n_procs = args.n_procs
    eval_n_jobs = args.eval_n_jobs
    sr_fair_cv = args.sr_fair_cv
    fixed_cv = args.fixed_cv
    cv_repeats = args.cv_repeats
    op_switch_interval = args.op_switch_interval
    disable_op_switch = args.disable_op_switch
    adaptive_switch = args.adaptive_switch
    switch_base_interval = args.switch_base_interval
    switch_min_interval = args.switch_min_interval
    switch_max_interval = args.switch_max_interval
    switch_window = args.switch_window
    switch_patience = args.switch_patience
    switch_ir_thresh = args.switch_ir_thresh
    switch_sr_thresh = args.switch_sr_thresh
    switch_deltaD_thresh = args.switch_deltaD_thresh
    switch_cooldown = args.switch_cooldown
    aos_enable = args.aos_enable
    aos_endpoint = args.aos_endpoint
    aos_model = args.aos_model
    aos_timeout = args.aos_timeout
    aos_max_retries = args.aos_max_retries
    aos_include_images = args.aos_include_images
    aos_debug = args.aos_debug
    aos_strict = args.aos_strict
    aos_init = args.aos_init

    if cfgs is not None:
        ac = cfgs.get("algo_config", {})
        ga = ac.get("ga", {})
        pop_size = int(ga.get("pop_size", pop_size))
        generations = int(ga.get("generations", generations))
        cv = int(ga.get("cv", cv))
        alpha = float(ga.get("alpha", alpha))
        scoring = ga.get("scoring", scoring)
        classifier = ga.get("classifier", classifier)
        sr_fair_cv = bool(ga.get("sr_fair_cv", sr_fair_cv))
        fixed_cv = bool(ga.get("fixed_cv", fixed_cv))
        try:
            cv_repeats = int(ga.get("cv_repeats", cv_repeats))
        except Exception:
            pass
        try:
            n_procs = int(ga.get("n_procs", args.n_procs))
        except Exception:
            n_procs = args.n_procs
        try:
            eval_n_jobs = int(ga.get("eval_n_jobs", args.eval_n_jobs))
        except Exception:
            eval_n_jobs = args.eval_n_jobs
        rates = ac.get("operator_rates", {})
        cxpb = float(rates.get("cxpb", cxpb))
        mutpb = float(rates.get("mutpb", mutpb))
        sw = ac.get("switching", {})
        mode = (sw.get("mode") or "fixed").lower()
        if mode == "fixed":
            op_switch_interval = int(sw.get("interval", op_switch_interval))
            disable_op_switch = False
            adaptive_switch = False
        elif mode == "adaptive":
            adaptive_switch = True
            asw = sw.get("adaptive", {})
            switch_base_interval = int(asw.get("base_interval", switch_base_interval))
            switch_min_interval = int(asw.get("min_interval", switch_min_interval))
            switch_max_interval = int(asw.get("max_interval", switch_max_interval))
            switch_window = int(asw.get("window", switch_window))
            switch_patience = int(asw.get("patience", switch_patience))
            switch_ir_thresh = float(asw.get("ir_thresh", switch_ir_thresh))
            switch_sr_thresh = float(asw.get("sr_thresh", switch_sr_thresh))
            switch_deltaD_thresh = float(asw.get("deltaD_thresh", switch_deltaD_thresh))
            switch_cooldown = int(asw.get("cooldown", switch_cooldown))
        aos = ac.get("aos", {})
        aos_enable = bool(aos.get("enabled", aos_enable))
        aos_endpoint = aos.get("endpoint", aos_endpoint)
        aos_model = aos.get("model", aos_model)
        # Allow timeout and retries to be configured via config
        try:
            aos_timeout = float(aos.get("timeout", aos_timeout))
        except Exception:
            pass
        try:
            aos_max_retries = int(aos.get("max_retries", aos_max_retries))
        except Exception:
            pass
        aos_include_images = bool(aos.get("include_images", aos_include_images))
        aos_debug = bool(aos.get("debug", aos_debug))
        aos_strict = bool(aos.get("strict", aos_strict))
        aos_init = bool(aos.get("init_on_start", aos_init))

    best_mask, best_fit, logbook, hof = run_ga(
        X=data.X,
        y=data.y,
        feature_names=data.feature_names,
        classifier=classifier,
        scoring=scoring,
        cv=cv,
        alpha=alpha,
        pop_size=pop_size,
        generations=generations,
        cxpb=cxpb,
        mutpb=mutpb,
        init_prob=args.init_prob,
        seed=args.seed,
        n_procs=n_procs,
        n_jobs_eval=eval_n_jobs,
        fixed_cv=fixed_cv,
        cv_repeats=cv_repeats,
        op_switch_interval=op_switch_interval,
        disable_op_switch=disable_op_switch,
        sr_fair_cv=sr_fair_cv,
        adaptive_switch=adaptive_switch,
        switch_base_interval=switch_base_interval,
        switch_min_interval=switch_min_interval,
        switch_max_interval=switch_max_interval,
        switch_window=switch_window,
        switch_patience=switch_patience,
        switch_ir_thresh=switch_ir_thresh,
        switch_sr_thresh=switch_sr_thresh,
        switch_deltaD_thresh=switch_deltaD_thresh,
        switch_cooldown=switch_cooldown,
        output_dir=args.output,
        aos_enable=aos_enable,
        aos_endpoint=aos_endpoint,
        aos_model=aos_model,
        aos_api_key=aos_api_key,
        aos_timeout=aos_timeout,
        aos_max_retries=aos_max_retries,
        aos_include_images=aos_include_images,
        aos_debug=aos_debug,
        aos_strict=aos_strict,
        aos_init=aos_init,
    )

    # Compute unpenalized downstream metric for the best mask using the SAME scoring used in GA
    clf = make_classifier(classifier)
    cv_override = None
    if fixed_cv:
        base_seed = int(args.seed if args.seed is not None else 0)
        if cv_repeats and int(cv_repeats) > 1:
            cv_override = RepeatedStratifiedKFold(n_splits=cv, n_repeats=int(cv_repeats), random_state=base_seed)
        else:
            cv_override = StratifiedKFold(n_splits=cv, shuffle=True, random_state=base_seed)
    cv_mean, cv_std, fold_scores = cv_scores_for_mask(
        best_mask, data.X, data.y, clf, scoring, cv, args.seed, cv_override=cv_override
    )

    # Augment best_solution.json with downstream metrics and penalty components
    os.makedirs(args.output, exist_ok=True)
    best_json_path = os.path.join(args.output, "best_solution.json")
    # Ensure base file exists
    save_results(args.output, best_mask, best_fit, logbook, data.feature_names)
    with open(best_json_path, "r") as f:
        best_data = json.load(f)
    selected_frac = best_data["num_selected"] / max(1, best_data["total_features"])
    best_data.update({
        "scoring": scoring,
        "cv_folds": cv,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_fold_scores": fold_scores,
        "alpha": alpha,
        "penalty": alpha * selected_frac,
        "selected_fraction": selected_frac,
    })
    with open(best_json_path, "w") as f:
        json.dump(best_data, f, indent=2)

    # Plot best fitness trend and diagnostics
    plot_best_fitness(logbook, args.output)
    plot_diversity(logbook, args.output)
    plot_improvement_rate(logbook, args.output, ema_alpha=args.ema_alpha_ir)
    plot_operator_success(logbook, args.output, ema_alpha_sr=args.ema_alpha_sr, ema_alpha_counts=args.ema_alpha_counts)
    plot_overview(logbook, args.output, ema_alpha_ir=args.ema_alpha_ir, ema_alpha_sr=args.ema_alpha_sr, ema_alpha_counts=args.ema_alpha_counts)

    selected_count = sum(best_mask)
    print(f"Best fitness (penalized): {best_fit:.4f}")
    if cv_mean == cv_mean:  # check not NaN
        print(f"Downstream {scoring}: mean={cv_mean:.4f}, std={cv_std:.4f}")
    print(f"Selected {selected_count}/{len(best_mask)} features. Results saved to '{args.output}'.")


if __name__ == "__main__":
    main()
