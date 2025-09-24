## XGBoost Ranker: Fixes and Enhancements

### TO-DO

Fixes (Recommended next)
- [ ] [HIGH] 4) Add early stopping for final model (recent-day validation slice)
- [ ] [HIGH] 9) Assert group contiguity and strict date sorting
- [ ] [HIGH] Spread sanitization (ask < bid; recompute/cap `spread_pct`)
- [ ] [HIGH] Drop weak days by health checks (min rows/day, min target coverage)
- [ ] [HIGH] Consistent feature projection (track `selected_features` and project at eval)
- [ ] [MEDIUM] 12) Resource control (PYTHONHASHSEED/OMP/MKL) and logging
- [ ] [MEDIUM] 15) Defensive metric computation with fallbacks (use unified evaluator)
- [ ] [MEDIUM] 7) Reproducibility (log versions/seeds)

Conditional (only if using CV/HPO)
- [ ] [MEDIUM] 5) Day-boundary-safe CV with purge (FixedPurgedKFold)
- [ ] [HIGH] 11) Leak-free per-fold preprocessing; monotone vector length safety

Enhancements (Recommended next)
- [ ] 19) Dry-run mode and acceptance gating (NDCG/IC/Precision)
- [ ] 8) CLI/config switches (tsfresh toggle, scale mode, lookahead, K, val-days)
- [ ] 9) Save training metadata and version pins with model artifacts

Enhancements (Defer / Later)
- [ ] 10) Expand causal rolling feature set (returns/vol/volume/IV slopes)
- [ ] 11) Performance hygiene (float32, n_jobs)
- [ ] 12) GPU support/predictor toggle
- [ ] 13) Sparse-aware preprocessing for categorical OHE
- [ ] 14) Memory-conscious parquet ingestion and dtype downcasting
- [ ] 16) Slice analytics and comprehensive Markdown reporting
- [ ] 17) Winsorization after HPO using split-tags (then transform val/test)
- [ ] 18) Exponential gains for NDCG

### IN-PROGRESS
- [ ] None

### DONE
- [x] Baseline snapshot: `experiments/step0_baseline` and workflow README
- [x] Experiments scaffolding and stepwise plan (experiments/)
- [x] [CRITICAL] 1) Raw-value computations for engineered features and Sharpe/returns target
- [x] [CRITICAL] 2) Eliminate tsfresh lookahead leakage (replace with causal rolling features)
 - [x] [CRITICAL] 8) Daily rank labels from realized forward returns with coverage gates — Implemented and evaluated (step3). Result: lower OOS NDCG vs step2; kept as a documented variant but not adopted as default.
 - [x] [HIGH] 3) Remove double-scaling (drop `StandardScaler` from model pipeline) — Implemented in `experiments/step4_no_scaling`. Result: large OOS NDCG degradation (−28–33%) and Spearman sign issues vs step2. Not adopted; keep current scaling path (step2) for this dataset.

This document captures targeted improvements for `train_xgboost_ranking_model.py` and the preprocessing/evaluation pipeline. Items are grouped as Fixes vs Enhancements, with priority labels: [CRITICAL], [HIGH], [MEDIUM].

---

## Fixes

### [CRITICAL] 1) Compute returns/Sharpe and engineered features on raw values (not scaled)
- **Problem**: `utils.preprocess_data` may scale numeric columns (per `config.yaml`). Step 3.5 then computes `price_change_1d` as `(last_t - last_{t-1}) / last_{t-1}` and Step 4 computes Sharpe from `last`. If `last` is scaled, these are computed on scaled values, distorting returns and Sharpe.
- **Impact**: Mis-specified supervision signal; unstable magnitudes; degraded generalization.
- **Fix (A - preferred)**: Compute `price_change_1d`, `iv_change_1d`, and Sharpe target on raw columns before any scaling. Preserve raw copies in `utils` (e.g., `last_raw`, `implied_volatility_raw`) and use those in training steps.
- **Fix (B)**: Exclude `last` and `implied_volatility` from `numerical_cols_to_scale` in `config.yaml` so they remain raw.
- **Code sketch**:

```python
# In utils._scale_data, before scaling:
raw_cols_to_preserve = ['last', 'implied_volatility']
for c in raw_cols_to_preserve:
    if c in df.columns:
        df[f'{c}_raw'] = df[c]

# In training Step 3.5 and target calc:
last_col = 'last_raw' if 'last_raw' in df.columns else 'last'
iv_col = 'implied_volatility_raw' if 'implied_volatility_raw' in df.columns else 'implied_volatility'
```

### [CRITICAL] 2) Eliminate tsfresh lookahead leakage
- **Problem**: `_add_tsfresh_features` computes entity-level features over the full time series and merges back to all rows, introducing future information.
- **Impact**: Inflated CV metrics; worse out-of-sample behavior.
- **Fix (A - recommended)**: Replace tsfresh features with explicit rolling-window features computed causally per row (e.g., rolling mean/std/vol of `last_raw`).
- **Fix (B)**: If tsfresh is required, compute on rolling/sliced windows up to the current date only and merge aligned by date (complex and slower).
- **Code sketch (rolling features)**:

```python
df = df.sort_values(['contractID', 'date'])
window = 20
df['price_roll_mean_20'] = df.groupby('contractID')['last_raw'].transform(lambda s: s.rolling(window, min_periods=5).mean())
df['price_roll_std_20']  = df.groupby('contractID')['last_raw'].transform(lambda s: s.rolling(window, min_periods=5).std())
```

### [HIGH] 3) Remove double-scaling; trees do not require scaling
- **Problem**: Data is scaled in `utils` (RobustScaler) and again in the model pipeline via `StandardScaler`.
- **Impact**: Redundant transforms; potential signal distortion; slower fit/predict.
- **Fix**: In the model pipeline keep only `SimpleImputer` for numerics and remove `StandardScaler`. If other models need scaling, centralize it once.
- **Acceptance**: For the ranker path, set `numerical_cols_to_scale: []` in `config.yaml` so `utils` performs no dataset-wide scaling.
- **Location of scaling**: Keep any scaling inside the sklearn pipeline that is fit per fold/slice only; never apply a global scaler prior to splitting.

### [HIGH] 4) Add early stopping for the final model fit
- **Problem**: Final training uses fixed `n_estimators=1500` without early stopping.
- **Impact**: Overfitting risk; unnecessarily large models.
- **Fix**: Hold out a recent validation window (e.g., last 10 trading days). Provide `ranker__eval_set`/`ranker__eval_group` and `ranker__early_stopping_rounds`.
- **Code sketch**:

```python
# Split final X,y by date into train/val (recent days as val)
ranker = final_pipeline.named_steps['ranker']
ranker.set_params(early_stopping_rounds=50)
final_pipeline.fit(
    X_train[features], y_train,
    ranker__eval_set=[(X_val[features], y_val)],
    ranker__eval_group=[val_group_info],
    ranker__group=train_group_info
)
```

### [MEDIUM] 5) Ensure day-boundary-safe CV
- **Problem**: `TimeSeriesSplit` can split within a day, while ranking groups are daily.
- **Impact**: A day may be partially in train and val; valid but less pure grouping.
- **Fix**: Implement a date-level splitter that preserves whole-day boundaries across folds.
- **Upgrade (from production template)**: Use a purged time-aware CV (e.g., `FixedPurgedKFold`) that:
  - Splits by unique dates (`asof_date`),
  - Trains strictly on past data,
  - Applies a configurable purge/embargo window (e.g., `purge_days=5`) before validation to avoid leakage.

### [MEDIUM] 6) Warning cleanup & robustness tweaks
- **SettingWithCopyWarning**: Use `.loc` and explicit copies to avoid chained assignment.
- **Categorical `type`**: Cast to `str` consistently before OHE; ensure missing categories handled.
- **NaN/Inf guards**: Keep checks prior to predict (already present).

### [MEDIUM] 7) Reproducibility & seeds
- **Fix**: Set and persist seeds for numpy, xgboost (`seed`), and Optuna sampler; log in metadata.
- **Benefit**: Stable CV; comparable runs.

### [CRITICAL] 8) Daily rank labels from realized forward returns with coverage gates
- **Problem**: Using global Sharpe quantiles can misalign per-day ranking supervision; also, days with single rows produce meaningless ranks.
- **Fix**: Switch to realized forward returns for a shorter, high-coverage horizon (e.g., 3d). Build per-day labels via percentiles → ordinal bins, and drop days with <2 rows before label creation.
- **Details**:
  - Higher coverage at 3d vs 5d.
  - Per-day percentiles avoid cross-day distribution drift.
  - Enforce label integrity by removing single-row days.
- **Code sketch**:

```python
target_col = f'target_fwd_return_{h}d'
df_labeled = df.dropna(subset=[target_col]).copy()
sizes = df_labeled.groupby('asof_date')[target_col].size()
good_days = sizes[sizes >= 2].index
df_labeled = df_labeled[df_labeled['asof_date'].isin(good_days)]
pct = df_labeled.groupby('asof_date')[target_col].rank(method='first', pct=True)
df_labeled['rank_label'] = pd.cut(
    pct, bins=[0.0, 0.50, 0.75, 0.90, 0.97, 1.0], labels=[0,1,2,3,4], include_lowest=True
).astype(np.int32)
```

### [HIGH] 9) Group contiguity assertions and strict date-sorted order
- **Problem**: If groups (days) are not contiguous, ranking group sizes can mismatch transformed matrices.
- **Fix**: Assert contiguity on `asof_date` transitions; always sort by date before grouping and transformation.
- **Benefit**: Prevents silent metric corruption and hard-to-debug misalignments.

### [HIGH] 10) Proper group sample-weighting shape and validation
- **Problem**: Ranking sample weights must be one-per-group (date), not per-row; mismatches silently degrade training.
- **Fix**: Compute per-day weights, validate `len(weights) == len(groups)` and error on mismatch.
- **Benefit**: Stable, interpretable weighting that aligns with XGBoost's ranking API.

### [HIGH] 11) Leak-free per-fold preprocessing and monotone vector safety
- **Problem**: Fitting a single preprocessor across all CV folds leaks information; also, monotone constraint vectors can misalign with transformed features.
- **Fix**: Fit a fresh preprocessor per fold using only train-fold data; validate constraint vector length against transformed feature count and disable if mismatched.
- **Benefit**: Eliminates CV leakage; prevents constraint-induced failures across sklearn versions.
- **Acceptance**:
-  - No dataset-wide scaling in `utils` for the ranker path (`numerical_cols_to_scale` is empty).
-  - Preprocessor is fit on the train slice only and reused to transform val/test (no refits on val/test).
-  - Add a guard before training: `assert not any(c.endswith('_scaled') for c in X.columns)` to ensure no pre-scaled columns leak in.
-  - Verify monotone constraint vector length equals the transformed feature count; if mismatched, disable constraints for that run.

### [MEDIUM] 12) Determinism and resource control (threads/env)
- **Fix**: Set `PYTHONHASHSEED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and seeds for Python/NumPy; log package versions.
- **Benefit**: More reproducible CV and training on shared machines.

### [MEDIUM] 13) Enhanced input normalization (mapping, date coercion, derived columns)
- **Fix**: Robustly map heterogeneous column names to canonical schema; coerce dates to tz-naive datetimes; compute missing derived columns (e.g., yield curve slope, `spread_pct`).
- **Benefit**: Higher ingestion resilience across datasets; fewer runtime surprises.

### [HIGH] Additional fixes from production template
- **Spread sanitization (ask < bid, inconsistent `spread_pct`)**
  - Fix inverted quotes (swap bid/ask when `ask < bid`).
  - Recompute `spread_pct` from bid/ask mid and clip to a hard cap (e.g., 0.6).
  - Reduces noisy outliers that corrupt training/evaluation.
- **Drop weak days by health checks**
  - Enforce minimum rows/day (e.g., `min_rows_per_day=200`) and minimum target coverage per day (e.g., `min_target_coverage=0.8`).
  - Removes low-quality groups that skew ranking metrics.
- **Consistent feature projection**
  - Project both train and inference dataframes to a tracked feature set (`selected_features`), add missing cols with zeros.
  - Prevents shape drift and inference-time failures.

---

## Enhancements

### 8) Add CLI/config switches for experimentation
- **Goal**: Toggle behavior without code edits.
- **Flags**:
  - `--disable-tsfresh` (or `use_tsfresh: false` in config)
  - `--scale-mode {none,robust,standard}` to control pre-scaling globally
  - `--lookahead-days` to change Sharpe horizon
  - `--k-ndcg` to control evaluation @K consistently across CV/final
  - `--val-days` to size the final early-stopping window

### 9) Save training metadata and version pins with the model
- **Problem**: Unpickling errors across sklearn versions and lack of provenance.
- **Fix**: Save a JSON alongside the model: package versions, feature hash, params, date ranges, training timestamp, CV metrics summary.
- **Benefit**: Easier debugging, reproducibility, safe loading.

### 10) Expand feature set with causal, rolling signals
- **Additions**:
  - Rolling returns/volatility (e.g., 5/20/60-day)
  - Rolling OI/volume velocity and ratios
  - Rolling IV slope/skew; IV/price correlations
  - Regime-aware features (VIX state, spread tightness) computed causally
- **Constraint**: All features must be computed using only past information.

### 11) Performance hygiene
- **Keep `float32`** for numerics.
- **Parallelism**: Keep `n_jobs=-1` where safe; avoid over-subscription.
- **Trial budgets**: For Optuna, use `timeout` for exploration; lift for final tuning.

### Additional enhancements from production template
- **Winsorization and log1p features**
  - Fit winsor bounds per feature on train (e.g., 1st–99th percentile), apply to val/test for outlier robustness.
  - Add `log1p` transforms for heavy-tailed counts (volume, open interest) to stabilize variance.
- **Robust liquidity filters**
  - Apply OI and spread-based filters with flexible column aliases (e.g., `options_data_open_interest`, `open_interest`, `oi`).
  - Configurable thresholds (e.g., `liquidity_filter_oi`, `liquidity_filter_spread_pct`).
- **Sample weighting by group-level liquidity**
  - Compute per-day weights from average OI and spread; map to rows.
  - Emphasizes liquid conditions; can stabilize learning.
- **Monotone constraints**
  - Provide monotonic direction per feature (e.g., negative for spread-related features, positive for OI/volume).
  - Apply aligned constraint vector to `XGBRanker` to enforce domain priors.
- **Deterministic hyperparameter search with CV stability**
  - Grid-search over a compact param grid; compute CV NDCG@K mean ± std.
  - Select best only if std ≤ threshold (e.g., `max_cv_std=0.02`) to avoid unstable configs.
- **Quality gates for OOT acceptance**
  - Require minimum OOT NDCG, Rank IC (Spearman), and Precision@K (e.g., `min_oot_ndcg`, `min_oot_rank_ic`, `min_precision_at_50`).
  - Fail closed if gates are not met.
- **Memory-conscious dataset loading**
  - Chunked parquet ingestion, periodic `gc.collect()`, and optional `psutil` checks against `max_memory_gb`.
  - Scales to large datasets with bounded memory.
- **Feature engineering refinements**
  - Interactions: `delta*gamma`, `vega*IV`.
  - Time-decay rate: `-theta / (days_to_expiration + ε)`.
  - Intra-day percentile ranks for key columns (per `asof_date`).
- **Reports and artifacts structure**
  - Separate `artifacts/production_ranker` and `reports/production_ranker` directories for model assets and evaluation outputs.
- **Additional metrics**
  - Add Rank IC (Spearman) alongside NDCG/Precision/MAP for ranking quality from a correlation perspective.

### 12) GPU support and predictor selection
- **Enhancement**: Optional `enable_gpu` flag to use `gpu_hist` and `gpu_predictor` when CUDA available; default to `hist`/`auto` on CPU.
- **Benefit**: Faster HPO and final training at scale.

### 13) Sparse-aware preprocessing for large-scale OHE
- **Enhancement**: Use `OneHotEncoder(handle_unknown='ignore', sparse_output=True)` and `ColumnTransformer(..., sparse_threshold=1.0)` to keep union sparse.
- **Benefit**: Reduces RAM pressure on tens of millions of rows; XGBoost handles sparse inputs efficiently.

### 14) Memory-conscious parquet ingestion
- **Enhancement**: Chunked loading of partitioned parquet with periodic `gc.collect()` and dtype downcasting (`float64`→`float32`, `int64`→smaller ints).
- **Benefit**: Enables training on ~7M+ samples within constrained memory.

### 15) Defensive metric computation and fallbacks
- **Enhancement**: Prefer external `eval_utils` (if present) for grouped NDCG/Precision/RankIC; otherwise use internal guarded implementations.
- **Benefit**: Robust metrics under varied environments and signatures.

### 16) Slice analytics and comprehensive reporting
- **Enhancement**: Compute slice metrics (by DTE, |delta|, spread tightness, VIX regimes); generate Markdown report with acceptance table, artifacts list, versions, and summary.
- **Benefit**: Faster diagnosis of where the model excels or struggles; audit-ready documentation.

### 17) Winsorization after HPO using split-tags
- **Enhancement**: Fit winsor bounds on a concatenated train/val frame using a `__split` tag, then split back by tag (not by date) to avoid cross-date bleed.
- **Benefit**: Prevents subtle leakage while aligning distribution between train and val for final fit.

### 18) Exponential gains for NDCG objective
- **Enhancement**: Use `ndcg_exp_gain=True` to align scoring with acceptance criteria that emphasize top ranks more heavily.
- **Benefit**: Improves focus on top-K ranking quality where it matters most.

### 19) Dry-run mode and acceptance gating
- **Enhancement**: CLI `--dry-run` to load, preprocess, apply health gates, and print stats without training; enforce OOT acceptance thresholds (NDCG, Rank IC, Precision@K) and emit a PASS/FAIL status.
- **Benefit**: Faster iteration and safer production rollouts.

---

## Notes specific to the "Production-Ready Ranker with Critical Fixes" variant

- **Weighting semantics (instance vs group)**
  - This variant uses per-instance `sample_weight` derived from per-day liquidity. That is acceptable with `XGBRanker`'s sklearn API. If true group-level weighting is desired, consider the native XGBoost API (support for `group_weight`) or broadcast day-level weights to all rows in the day consistently.

- **Stable sorting for contiguity**
  - Prefer `kind='mergesort'` when argsorting indices by date to ensure stability across equal keys. Use consistent stable sorts in train/val/test ordering.

- **Categorical handling**
  - Although `OneHotEncoder` is imported, the path bypasses a preprocessing pipeline. If categorical features are present, provide an optional preprocessing stage with sparse OHE and keep it leak-free per CV fold.

- **Unused imports & warnings**
  - Clean up unused imports (e.g., `StandardScaler`, `psutil` if not used) and minimize global warning filters. Favor targeted warning suppression around known third-party deprecations.

- **Memory guardrails**
  - If adopting `psutil`, add a periodic memory check against `max_memory_gb` to pause/abort or reduce concurrency before OOM during large parquet loads or grid search.

- **Metric definitions**
  - Precision@K in this variant uses a threshold derived from top-k labels per group. For graded labels, consider a consistent relevance rule (e.g., label ≥ 3) or switch to `precision_at_k_grouped` from shared eval utils for consistency.

- **NDCG gains**
  - Consider enabling exponential gains (`ndcg_exp_gain=True`) to emphasize correctness at top ranks if that matches business acceptance criteria.

---

## Suggested Implementation Phases

1) Phase 1 — Critical fixes
- [CRITICAL] Raw-value computations for engineered features and Sharpe target
- [CRITICAL] Eliminate tsfresh lookahead leakage

2) Phase 2 — High/Medium fixes
- [HIGH] Remove pipeline `StandardScaler`
- [HIGH] Add early stopping for final model
- [MEDIUM] Day-boundary-safe CV
- [MEDIUM] Warning cleanup & seeds

3) Phase 3 — Enhancements
- CLI/config flags; metadata/version pins
- Causal rolling feature expansion
- Performance hygiene

Each phase is incremental and independently testable via the existing evaluation script.


