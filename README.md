GA Feature Selection with DEAP

Overview
- Implements a binary genetic algorithm for feature selection using DEAP.
- Evaluates feature subsets via cross-validation with scikit-learn classifiers.
- Supports CSV datasets or built-in datasets (breast_cancer, iris, wine).

Quick Start
1) Install dependencies:
   - Python 3.8+
   - pip install -r requirements.txt

2) Run with a built-in dataset (no CSV needed):
   - python feature_selection_ga.py --sklearn-dataset breast_cancer --classifier logistic --cv 5 --alpha 0.02 --pop-size 60 --generations 40 --seed 42

3) Run with your CSV data:
   - python feature_selection_ga.py --csv your_data.csv --target-col target --classifier rf --cv 5 --alpha 0.02 --pop-size 80 --generations 50

 4) Run with an OpenML dataset (requires network):
    - By name:
      python feature_selection_ga.py --openml-name credit-g --openml-version 1 --classifier logistic --cv 5 --alpha 0.02
    - By ID:
      python feature_selection_ga.py --openml-id 1461 --classifier rf --cv 5 --alpha 0.02
    - Optional cache directory for downloads: add --data-home .openml_cache

Inputs
- CSV mode: Provide --csv path and --target-col name. All other columns are treated as features.
- Built-in datasets: --sklearn-dataset breast_cancer|iris|wine.
 - OpenML datasets: --openml-name NAME or --openml-id ID (optionally --openml-version and --data-home). If the dataset’s target is ambiguous, you can override with --target-col when the returned frame includes it.

Key Options
- --classifier logistic|svm|rf: Choose classifier for evaluation.
- --scoring: Any valid sklearn scoring (default: accuracy). Examples: f1_macro, roc_auc_ovr.
- --alpha: Sparsity penalty; higher favors fewer features.
- --pop-size, --generations, --cxpb, --mutpb: GA hyperparameters.
- --init-prob: Initial probability a feature is selected (default 0.1).
- --seed: Reproducibility.

Outputs
- Creates directory (default: ga_results) with:
  - best_solution.json: Best fitness, selected feature indices and names.
- evolution_log.csv: Per-generation statistics (avg, std, min, max fitness) and `diversity` (average pairwise Hamming distance fraction across the population).
  - Also includes `improvement_rate` (IR_t) defined as `(avg_t - avg_{t-1}) / avg_{t-1}`; at gen 0 it's NaN; if the previous avg is ~0, it's set to NaN.
  - Operator success rates: `sr_cx`, `sr_mut` (per-generation success rate of crossover and mutation), and counts `cx_applied`, `mut_applied`.
  - Stagnation counter: `stagnation` = number of consecutive generations without change in best fitness (max). Resets to 0 when best improves.
  - best_solution.json also includes downstream CV metrics: `scoring`, `cv_folds`, `cv_mean`, `cv_std`, `cv_fold_scores`, and penalty components `alpha`, `penalty`, `selected_fraction`.
  - best_fitness.png: Line plot of per-generation best fitness (requires matplotlib; auto-generated after run).

Notes
- Empty selections are repaired by forcing at least one feature on.
- For linear/Kernel methods, data is scaled via StandardScaler in a Pipeline.
- If a subset causes the model to fail, a large negative fitness is applied.

Visualization Metrics
- Population Diversity (Hamming): For a population of size N and bit-length L, we compute
  diversity = (1 / (L * N * (N - 1))) * sum_j (2 * n1_j * (N - n1_j)), where n1_j is the count of ones on bit j.
  This equals the average pairwise Hamming distance normalized by L, ranging in [0, 1]. Lower values indicate loss of diversity and potential premature convergence.
- Fitness Improvement Rate (IR): IR_t = (avg_t - avg_{t-1}) / avg_{t-1}, where avg_t is the generation t mean fitness from the log. Useful to detect stagnation/plateaus. For t=0, or when avg_{t-1}≈0, IR is recorded as NaN.
- Operator Success Rate (SR): For an operator (e.g., crossover or mutation), SR = (# of offspring whose fitness > baseline) / (# operator applications). Baselines:
  - Crossover: baseline is max(parent1_fitness, parent2_fitness). Two children counted per applied crossover.
  - Mutation: baseline is the individual's fitness immediately before mutation. 为避免交叉+变异重复归因，本实现仅统计当代未经历交叉的个体的变异成功率。
 - Stagnation Detection: S_t = S_{t-1} + 1 if f_best,t == f_best,t-1 else 0. We compare generation-wise best fitness with a tight tolerance; when the best improves, S_t resets to 0.

Tips
- Increase --alpha to push for sparser subsets.
- Increase --generations and/or --pop-size for better search at the cost of runtime.
- Use --seed to compare runs fairly.

Baseline Comparison
- Compare GA against common selectors under the same CV/scoring/alpha:
  - python compare_baselines.py --sklearn-dataset breast_cancer --classifier logistic --cv 5 --alpha 0.02 --seed 42 \
      --baselines all,random,kbest_f,kbest_mi,rf_topk,l1_logistic,lasso,rfe_logistic,rfe_rf \
      --random-iters 200 --k-grid auto --C-grid 0.01,0.1,1,10
- Outputs `baseline_results/` with:
  - baseline_summary.csv: One row per method with best penalized fitness, downstream CV mean/std, selected count, and features.
  - best_<method>.json: Best subset per method.
- Notes:
  - Penalized fitness = CV score (with chosen classifier) minus `alpha * (k/d)` to match GA's objective.
  - `kbest_f/kbest_mi` sweep k over an automatic grid (`--k-grid auto`) or a custom list (e.g., `--k-grid 5,10,15`).
  - `random` samples `--random-iters` masks with `--init-prob` ones probability.
  - `l1_logistic` (aka LASSO-style sparse logistic) tries a small `C` grid; if no features survive, it falls back to the strongest one.
  - `lasso` is an alias of `l1_logistic` for clarity.
  - `rfe_logistic`/`rfe_rf` run RFE with logistic (L2) or random forest estimators over the same k-grid; final evaluation uses the unified penalized fitness.

Baselines Explained
- all: Use all features; good upper bound on raw model capacity, but penalized by alpha.
- random: Random masks; acts as a naive search baseline under the same sparsity prior.
- kbest_f: Univariate F-test (ANOVA) ranking; pick top-k. Fast, ignores feature interactions.
- kbest_mi: Univariate mutual information ranking; non-linear dependence, still ignores interactions.
- rf_topk: Rank by RandomForest importances; leverages interactions/non-linearity but can favor correlated groups.
- l1_logistic (lasso): L1-penalized logistic regression induces sparse weights; SelectFromModel keeps non-zero coefficients. Grid over C controls sparsity.
- rfe_logistic: Recursive Feature Elimination using logistic regression as estimator; repeatedly removes least important features to reach k.
- rfe_rf: RFE using RandomForest as estimator; relies on tree-based importances during elimination.

Configuration-Driven Runs (Recommended)
- Place runtime configs under `config/` and start the run with a single flag:
  - python feature_selection_ga.py --use-config --seed 42
- Files and purpose:
  - `config/task_info.json` — dataset + task metadata
    - Example (sklearn breast_cancer):
      {
        "dataset_name": "sklearn:breast_cancer",
        "dataset_source": "sklearn",
        "dataset_size": "n=569",
        "num_features": 30,
        "classification_model": "logistic",
        "scoring": "accuracy",
        "cv_folds": 5,
        "fitness_function": "CV_mean - alpha * (k/d)",
        "alpha": 0.02
      }
  - `config/operator_pools.json` — operator pools with parameter schema (for AOS prompts)
    - Selection: tournament (params: k), sus, best
    - Crossover: one_point, two_point, uniform (params: prob)
    - Mutation: flip_bit (params: prob), uniform_int (params: prob, low, up)
  - `config/algo_config.json` — GA hyperparams, switching strategy, AOS options, paths
    - ga: {pop_size, generations, cv, alpha, scoring, classifier, sr_fair_cv}
    - operator_rates: {cxpb, mutpb}
    - operators.current: current operator names + params
    - switching:
      - mode: "fixed" or "adaptive"
      - interval: integer (fixed mode); set 0 to disable switching
      - adaptive: {base_interval, min_interval, max_interval, window, patience, ir_thresh, sr_thresh, deltaD_thresh, cooldown}
    - aos: {enabled, endpoint, model, include_images}
    - paths: {overview_image}

Operator Switching and AOS
- Fixed vs Adaptive switching
  - Fixed: switches every `interval` generations (no switch on the final generation)
  - Adaptive (exponential backoff-style): interval increases on persistent ineffectiveness (by IR/SR/ΔD window), recovers toward base on effectiveness
- SR fairness: enable `sr_fair_cv` to compare parents/offspring under a generation-fixed CV splitter (reduces noise in SR)
- AOS (LLM-driven operator selection)
  - Enable with `config/algo_config.json` → `"aos": {"enabled": true, ...}` (or `--aos-enable`)
  - At each switch point:
    1) Refresh overview.png
    2) Summarize state (v2 prompt). If `include_images=true`, overview.png is attached as base64
    3) Ask for operator decision (v2 prompt). Expected JSON (v2 format):
       {
         "cxpb": 0.7,
         "mutpb": 0.3,
         "Selection": {"name": "tournament", "parameter": {"k": 3}},
         "Crossover": {"name": "uniform", "parameter": {"prob": 0.5}},
         "Mutation": {"name": "flip_bit", "parameter": {"prob": 0.033}}
       }
    4) Bind decision to DEAP:
       - tournament(k) → selTournament(tournsize=k)
       - uniform(prob) → cxUniform(indpb=prob)
       - flip_bit(prob) → mutFlipBit(indpb=prob | 1/n_features if 0)
       - uniform_int(prob, low, up) → mutUniformInt(low, up, indpb=prob | 1/n_features if 0)
  - If AOS is disabled (or request fails), switching falls back to a small random operator pick from the internal pool.

Logging (evolution_log.csv)
- Per-generation fields include:
  - Fitness stats: gen, nevals, avg, std, min, max
  - Diagnostics: diversity, improvement_rate, stagnation, sr_cx, sr_mut, cx_applied, mut_applied
  - Operator names: op_sel, op_cx, op_mut
  - Operator params (JSON string): op_sel_param, op_cx_param, op_mut_param
    - tournament: {"tournsize": k}
    - uniform crossover: {"indpb": prob}
    - flip_bit: {"indpb": prob}
    - uniform_int: {"low": int, "up": int, "indpb": prob}
  - Rates: cxpb, mutpb
  - Switching: switch_interval, since_last_switch, switch_effective

Notes & Tips
- Final generation does not trigger switching (avoids wasted work)
- Avoiding tight_layout warnings: complex figures use manual layout; no functional impact
- SR fairness (`--sr-fair-cv`): reduces noise in SR by comparing parents/offspring with the same CV splitter in a generation
- To disable switching entirely: set `config/algo_config.json` → `switching.mode="fixed"` and `switching.interval=0`
- To use only AOS switching (no random fallback on failure): tell us if you want a `fallback_on_failure: false` flag; we can add it so switching is skipped if AOS fails

LLM Endpoint Live Test
- Provide endpoint/model/api key via env vars and run:
  - export AOS_API_KEY=...
  - python tests/aos_live_test.py
- Prints the exact v2 prompts and raw JSON responses for both state summary and operator decision; then prints a normalized decision (validated & clamped)

Debugging & New Utilities
- AOS debug printing (prompts + responses)
  - Enable via config: `config/algo_config.json` → `"aos": {"debug": true}` (or CLI `--aos-debug`).
  - At each switch point prints:
    - `[AOS] Switch at gen=G (interval=K, adaptive_switching=..., aos_enabled=...)`
    - `[AOS][REQUEST] summarize_state messages:` (System + User prompts; if `include_images=true`, also prints attached image path)
    - `[AOS][RESPONSE] summarize_state raw content:` (model JSON as-is)
    - `[AOS][REQUEST] choose_operators messages:` (System + User prompts)
    - `[AOS][RESPONSE] choose_operators raw content:` (model JSON as-is)
    - `[AOS][POST] normalized decision:` (validated v2 JSON; warnings when values are clamped)

- Stable parameter logging per generation
  - `evolution_log.csv` now includes operator parameters and rates every generation:
    - `op_sel_param`, `op_cx_param`, `op_mut_param` as JSON strings
    - `cxpb`, `mutpb`
  - Examples:
    - tournament: `{ "tournsize": 3 }`
    - uniform crossover: `{ "indpb": 0.5 }` (LLM `prob` → DEAP `indpb`)
    - flip_bit: `{ "indpb": 0.0333 }` (LLM `prob`; when `prob=0` falls back to `1/n_features`)
    - uniform_int: `{ "low": 0, "up": 1, "indpb": 0.0333 }`

- Overview refresh at switch points
  - Before each switch, the latest `overview.png` is regenerated to reflect the most recent generations.
  - If `include_images=true`, summarize_state attaches the refreshed overview image as base64.

- Layout warnings removed
  - Complex figures (operator_success, overview) use manual layout and bbox saving, avoiding `tight_layout` warnings.

- Final generation no switching
  - Switching is suppressed at `gen==generations` to avoid wasted work on the last generation.

Prompt Format (v2) and Parameter Mapping
- Decision JSON schema (must match operator pools):
  {
    "cxpb": 0.7,
    "mutpb": 0.3,
    "Selection": {"name": "tournament", "parameter": {"k": 3}},
    "Crossover": {"name": "uniform", "parameter": {"prob": 0.5}},
    "Mutation": {"name": "flip_bit", "parameter": {"prob": 0.033}}
  }
- Binding to DEAP (prob → indpb):
  - tournament(k) → selTournament(tournsize=k)
  - uniform(prob) → cxUniform(indpb=prob)
  - flip_bit(prob) → mutFlipBit(indpb=prob | 1/n_features if 0)
  - uniform_int(prob, low, up) → mutUniformInt(low, up, indpb=prob | 1/n_features if 0)

Troubleshooting
- Fixed vs Adaptive switching
  - `switching.mode="fixed"` uses `interval` directly (first switch at `gen%interval==0`).
  - `switching.mode="adaptive"` uses `adaptive.base_interval` as initial; interval adjusts via backoff/recover.
  - If you expect adaptive to trigger at `gen=base`, we can switch to a `last_switch_gen`-based trigger logic (contact us).
- AOS enabled but still random switching?
  - Ensure `aos.enabled=true` and the switch mode actually triggers (fixed `interval>0` or adaptive window reaches interval).
  - Network/API issues will fallback to a small random operator pick; enable `aos.debug=true` and check logs.
- Disable switching entirely
  - Set `switching.mode="fixed"` and `switching.interval=0`.

Visualization
- Best fitness curve (per generation):
  - After running GA, plot the best fitness from the log:
    - python plot_ga_metrics.py --log-csv ga_results/evolution_log.csv --out ga_results/best_fitness.png
  - Requires matplotlib: pip install matplotlib
- Diversity curve (per generation) with threshold line:
  - The GA script auto-saves `ga_results/diversity.png` after each run (horizontal line = historical mean).
  - You can also generate manually or customize threshold:
    - python plot_diversity.py --log-csv ga_results/evolution_log.csv --out ga_results/diversity.png
    - Add `--no-hline` to hide the threshold, or `--hline-value 0.2` to set a custom line.
 - Improvement rate (IR) per generation:
   - The GA script auto-saves `ga_results/improvement_rate.png` after each run.
   - Encoding: bars colored green (positive), red (negative), gray (zero/NaN); includes a line overlay.
 - Operator success rates per generation:
   - The GA script auto-saves `ga_results/operator_success.png` (multi-line chart for `sr_cx` and `sr_mut`).
 - Overview:
   - The GA script auto-saves `ga_results/overview.png`, a 2x2 dashboard combining Best Fitness, Diversity, IR, and Operator SR+Counts.
