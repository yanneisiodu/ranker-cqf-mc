# Neural Ranker: Complete Development History & Current State

## What This System Does

Ranks ~6,500 daily SPY stock options by expected 5-day return using a Transformer neural network. The top-ranked options are filtered through a selective meta-allocator that decides which trades to accept/reject, then positions are managed by a causal simulation engine with take-profit, stop-loss, and trailing-stop exit strategies.

## Architecture (Active Code Only)

```
Daily SPY option chain (~6,500 options after liquidity filter)
    │
    ▼
[ChainTransformer] — full O(n²) attention over the chain
    │                  embed_dim=256, n_heads=4, n_layers=2
    │                  1.25M params, ListMLE loss
    │
    ▼
Ranking scores per option
    │
    ▼
[Selective Meta-Allocator] — separate call/put XGBoost meta-models
    │                         predict P(good trade) per candidate
    │                         accept/reject individually
    │
    ▼
[Simulation Engine] — event-driven, causal settlement
    │                   take-profit, stop-loss, trailing stop
    │                   capital reserved on entry, freed on exit
    │
    ▼
Trade execution + P&L
```

## Active Files

| File | Purpose |
|------|---------|
| `neural_ranker.py` | ChainTransformer model, ListMLE loss, NDCG metric |
| `train_neural_ranker.py` | Training loop, DailyChainDataset, evaluate function |
| `causal_backtest.py` | Event-driven backtest using SimulationEngine |
| `simulation_engine.py` | Causal settlement, exit strategy, MaturedHistoryQueue |
| `build_candidate_dataset.py` | Builds per-trade meta-labeling dataset with maturity queue |
| `train_selective_operator.py` | Trains call/put XGBoost meta-models |
| `optuna_neural_sweep.py` | Hyperparameter tuning on Cloud Run |
| `cloud_config.py` | Cloud Run env var configuration |
| `config_tuned.yaml` | Canonical config (execution, risk, exit_strategy, splits) |
| `utils.py` | Data loading, feature engineering, purged splits |
| `logger.py` | Logging setup |
| `entrypoint.sh` | Cloud Run entrypoint (train + optuna modes) |
| `Dockerfile` | Container for Cloud Run GPU training |

## Data

**Source:** ThetaData (professional options data provider)
**Location:** `/Users/chinonsoisiodu/Documents/Projects/ranker_cqf_mc/Data/`
**Pre-processed parquets on GCS:** `gs://neural-ranker-training-data/parquet/`

**Files:** `year_2018_data.csv` through `year_2026_data.csv`
**Per year:** ~1-2M rows, ~7,000-9,000 options per day

**63 features per option:**
- First-order Greeks: delta, gamma, theta, vega, rho
- Higher-order Greeks (ThetaData): vanna, charm, vomma, speed, zomma, color, ultima, lambda_greek
- Price/liquidity: bid, ask, mid_price, volume, open_interest, relative_spread
- Option structure: strike, days_to_exp, moneyness, implied_volatility, type
- Market: spy_d_close, spy_d_sma_50, spy_d_rsi, spy_d_macd_hist, vix_d_close, spy_momentum
- Realized vol: realized_vol_5d, realized_vol_20d, realized_vol_60d
- VRP: vrp_20d (IV - realized_vol_20d)
- Engineered: rolling means/stds, cross-sectional ranks, etc.

**Target:** 5-day raw option return (buy at ask, sell at bid after 5 trading days)

**Data pipeline source:** `/Users/chinonsoisiodu/Documents/Projects/ML Trading/Data Ingest/spy_data/local_processing/spy_option_data_etl.py` — pulls from ThetaData API, joins with yfinance SPY/VIX and FRED treasury yields.

## Model Performance

### Best Ranker: NDCG@20 = 0.634
- Trained on 2018-2023, validated on 2024-2025
- Peaked at epoch 18
- Optuna-tuned: embed=256, heads=4, layers=2, dropout=0.25, lr=4.7e-4, weight_decay=2.3e-3

### XGBoost Comparison on Same Split
- NDCG@20 = 0.337 (neural ranker is nearly 2x better)

### NDCG at All K Levels (best model on 2024-2025)
| k | NDCG |
|---|------|
| @1 | 0.520 |
| @5 | 0.578 |
| @10 | 0.618 |
| @20 | 0.634 |

## Backtest Results

### CRITICAL: Non-Causal vs Causal Results

The non-causal backtest had three major bugs that inflated results ~900x:

1. **Same-day P&L booking:** 5-day returns were compounded daily instead of being held to exit
2. **Pre-matured efficacy features:** Meta-model features used outcomes not yet known (future leakage)
3. **Post-normalization spread filter:** Filtered on z-scored spread, not raw percentages

| Metric | Non-Causal (WRONG) | Causal (HONEST) |
|--------|-------------------|-----------------|
| Final equity from $10K | $445,917 | $492 |
| Total return | +4,359% | -95.1% |
| Win rate | 63% | 15.3% |
| Sharpe | 4.51 | -3.88 |

**The causal backtest is the ground truth.** All non-causal numbers in git history are unreliable.

### Causal Backtest Details (2024-2025)
- Exit strategy: 50% take-profit, 20% stop-loss, 15% trailing stop, 5-day max hold
- Only 2% of trades hit take-profit
- 41% hit stop-loss
- 45% held to max hold
- Puts dominated (294 vs 184 calls) with 11% put win rate

## What We Tried and Why It Failed

### Approaches That Didn't Improve NDCG

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| **Self-supervised pretraining** (masked reconstruction) | NDCG 0.496 (worse) | Reconstruction teaches wrong representations for ranking |
| **Multi-task training** (ranking + reconstruction) | NDCG 0.462 (worse) | Reconstruction loss dilutes ranking signal |
| **EWC fine-tuning** (2018-2023 base → 2024) | NDCG 0.568 (worse than base 0.591) | Fisher values too small, fine-tuning on 1 year insufficient |
| **Delta-hedged return target** | NDCG 0.288 (much worse) | Delta hedge degrades over 5 days, adds noise |
| **2-day horizon** | NDCG 0.437 (slightly worse) | Less signal, not more |
| **Training on more data** (2018-2024 vs 2018-2023) | NDCG 0.538 (worse) | Adding 2024 didn't help; model went stale on 2025 |
| **Hierarchical attention** (cluster by strike/expiry) | Built but untested | Superseded by causal fixes |
| **Warm-start ensemble** (3 experts + gate) | All peaked at epoch 1 | Fine-tuning degraded the base model; LR too high |
| **Regime-balanced ensemble** (from scratch) | E_recent 0.42, E_core 0.48 | Not enough data per expert to learn from scratch |

### Approaches That Worked

| Approach | Result |
|----------|--------|
| **Full attention on ~6.5K chain** (vs XGBoost top-200 pre-filter) | 99.8% recall of winners vs 15.2% |
| **Optuna hyperparameter tuning** | NDCG 0.58 → 0.634 |
| **ThetaData features** (higher-order Greeks, VRP) | XGBoost AUC 0.50 → 0.871 on meta-labeler |
| **Selective meta-allocator** (per-trade accept/reject) | Outperformed top-down day gate |
| **Causal simulation engine** | Revealed true performance |

### Key Architectural Decisions and Why

**Why full attention, not SetTransformer or pre-filtering:**
XGBoost's top-200 pre-filter loses 85% of actual winners. The transformer must see the full chain to learn cross-option patterns. Full O(n²) attention on ~6.5K options is feasible on L4 GPU (~48s/epoch) and MPS (~75min/epoch).

**Why ListMLE loss, not LambdaRank:**
ListMLE directly optimizes permutation likelihood. Research (DRO-ULTR, SIGIR 2025) shows ranking losses are fine; robustness comes from sample weighting, not loss function changes.

**Why bottom-up selective operator, not top-down day gate:**
Top-down: predict daily exposure → select trades = -67% return (failed)
Bottom-up: accept/reject each trade → exposure emerges = better architecture
The day gate can't see that specific puts at strike 484 with 9 DTE are about to expire worthless.

**Why raw-column filtering before normalization:**
Post-normalization `relative_spread <= 0.50` filters on z-scores, not raw percentages. This silently includes/excludes wrong options. Always filter on raw columns, then normalize for model input.

**Why trading-session purging, not calendar-day purging:**
`pd.Timedelta(days=5)` across a weekend only removes 3 trading sessions. Must purge by index into sorted unique trading dates.

## Cloud Run Infrastructure

**Artifact Registry:** `us-east4-docker.pkg.dev/gen-lang-client-0478886850/neural-ranker/neural-ranker-training:latest`
**GCS Bucket:** `gs://neural-ranker-training-data`
- `/data/` — raw CSVs
- `/parquet/` — pre-processed parquets (prepared, delta_hedged_5d, raw_2d, delta_hedged_2d)
- `/artifacts/` — trained model checkpoints

**GPU:** NVIDIA L4 (24GB VRAM), Cloud Run job in us-east4
**Timeout:** 1 hour max for GPU jobs
**Cost:** ~$1/hour

**Training speed:**
- L4 GPU: ~48s/epoch (without torch.compile, deterministic)
- L4 GPU: ~42s/epoch (with torch.compile, non-deterministic)
- Apple M5 MPS: ~75min/epoch (10x slower)

**Key env vars for Cloud Run:**
- `MODE`: train | optuna
- `TRAIN_YEARS`, `VAL_YEARS`: comma-separated
- `EPOCHS`, `PATIENCE`: training overrides
- `TORCH_COMPILE`: true/false (false for deterministic results)
- `TARGET_MODE`: net_long_return | net_delta_hedged_return
- `HORIZON_DAYS`: 5 (default) | 2

**Build and push:** `docker buildx build --platform linux/amd64 -t <image> --push .` (~3s for code-only changes)

## Causal Fixes Applied (CRITICAL — Do Not Revert)

These fixes are in the codebase. Any new code must follow the same rules:

1. **P&L settled on exit date only** — `SimulationEngine.step()` checks exit conditions daily, settles when triggered
2. **Capital reserved on entry, freed on exit** — `engine.open_position()` deducts from cash, exit returns to cash
3. **Efficacy features use only matured trades** — `MaturedHistoryQueue` in simulation_engine.py, pending queue in build_candidate_dataset.py
4. **Trading-session purging** — `PurgedWalkForwardSplit` uses index positions in sorted unique dates, not `pd.Timedelta`
5. **Date-level 3-way splits** — `split_train_cal_test_by_date()` ensures no date in multiple partitions
6. **Threshold tuning on calibration, not test** — `train_selective_operator.py` sweeps thresholds on cal data
7. **Raw-column liquidity filtering before normalization** — `filter_tradeable_raw()` from simulation_engine.py
8. **Config is single source of truth** — execution/risk/exit_strategy sections in config_tuned.yaml

## Current Status and What Needs to Happen Next

### The Core Problem
The ranker achieves NDCG@20 = 0.634 but the causal backtest shows -95% return. The model ranks well on paper but doesn't produce profitable trades under realistic execution with exit strategies.

### Possible reasons:
1. **Exit strategy parameters are wrong** — 50% take-profit may be too high for options (few trades reach it). 20% stop-loss may be too tight (options are volatile). Need to sweep exit params.
2. **The model picks too many puts** — 294 puts vs 184 calls, 11% put win rate. The model learned crash patterns from 2020/2022 training data.
3. **NDCG doesn't correlate with profitability under exit strategies** — ranking by 5-day terminal return is different from ranking by "which options hit 50% profit fastest."
4. **The model needs to be retrained with the causal fixes** — the liquidity filter change means the model was trained on a different universe than it's now tested on.

### Recommended next steps (in priority order):
1. **Sweep exit strategy parameters** — test different TP/SL/trailing combinations in causal_backtest.py
2. **Retrain the ranker with raw-column filtering** — the model was trained with post-normalization spread filter (wrong universe). Retrain on Cloud Run with the corrected `filter_tradeable_raw()`.
3. **Rebuild candidate dataset + retrain meta-models** — using maturity queue and corrected data
4. **Rerun causal backtest with selective operator** — using the corrected meta-models
5. **Monthly rolling retrain pipeline** — the model goes stale after 1-2 years. Build automated retraining.
6. **Hourly data** — another agent is working on ThetaData hourly pulls. Intraday chain dynamics could improve ranking.

### Research findings to apply:
- **Regime-Balanced Rolling Expert Ensemble** (from research agent) — rolling windows with recency decay, stress replay, deterministic gate. The architecture is sound but implementation needs warm-starting from a strong base + very low fine-tuning LR.
- **Selective prediction with conformal risk control** — the bottom-up accept/reject approach is correct per recent literature (SCoRE, SCRC, "When Alpha Breaks" paper)
- **Group-DRO over regime buckets** — add worst-regime weighting to ListMLE loss instead of changing the loss function
- **Monthly warm-started rolling retrain** — cheapest path to freshness ($3/retrain on Cloud Run)

## Files That Were Deleted and Why

All in git history if needed, but proven inferior:

- `realistic_backtest.py` — non-causal P&L booking, replaced by `causal_backtest.py`
- `selective_backtest.py` — non-causal, needs rewrite to use `simulation_engine.py`
- `operator_backtest.py` — top-down gate failed (-67% return)
- `train_operator.py` — trained failed top-down gate
- `build_operator_dataset.py` — built dataset for failed operator
- `pretrain.py` — self-supervised pretraining hurt NDCG
- `ewc_finetune.py` — EWC fine-tuning hurt NDCG
- `hierarchical_ranker.py` — built but never tested
- `sweep.py` — superseded by `optuna_neural_sweep.py`
- `regime_ensemble.py` — ensemble experts degraded from base
- `train_ensemble.py` — trained ensemble that didn't help
- `NEURAL_RANKER_PLAN.md` — superseded by this README
- `RESULTS.md` — superseded by this README

## BUGS FIXED (2026-03-31)

### Round 1 (causal fixes):
1. ✅ Trading-session purging (not calendar days)
2. ✅ Date-level 3-way splits with no overlap
3. ✅ Maturity queue — efficacy uses only settled outcomes
4. ✅ Threshold tuning on calibration, not test data
5. ✅ Canonical execution/risk/exit_strategy config in YAML

### Round 2 (code review bugs):
1. ✅ `train_neural_ranker_from_frames()` now filters BEFORE normalization
2. ✅ `optuna_neural_sweep.py` filters BEFORE normalization
3. ✅ `warmup_epochs`, `feature_noise`, `listmle_top_k` all wired through both training paths
4. ✅ ListMLE docstring corrected — truncated denominator documented
5. ⚠️ **Bug 5 (NDCG ceiling):** NOT YET FIXED — 5-bin relevance creates massive ties

### Round 3 (training consistency fixes):
1. ✅ Dead reconstruction code removed from `train_one_epoch`
2. ✅ `from_frames` and `from_datasets` now use same scheduler (warmup + cosine LambdaLR)
3. ✅ `from_datasets` `actual_config` now includes `listmle_top_k`, `warmup_epochs`, `feature_noise`
4. ✅ Optuna sweep filters on raw columns before normalization
5. ✅ `build_candidate_dataset.py` filters on raw columns before normalization
6. ✅ Unused `torch.nn` import removed

### Remaining:
- ⚠️ **Bug 5:** 5-bin relevance + tie-blind ListMLE = NDCG ceiling at ~0.63-0.66. Consider: more bins (10-20), continuous relevance, or ApproxNDCG loss.
- ✅ **Retrained ranker** on corrected universe: NDCG@20 = 0.601 (embed=256, honest number)
- ✅ Optuna re-sweep on corrected codebase (best 0.626 in sweep, 0.600 on full run)
- ✅ Wired selective meta-operator into causal_backtest.py (bucketed exits + regime-aware put filtering)
- ✅ Rebuilt candidate dataset with matured-only efficacy
- ✅ Retrained meta-models (call AUC 0.64, put AUC 0.50 with honest maturity delay)
- ✅ Causal backtest with full stack: $10K → $3,634 (-63.7%), 135 trades (calls only), 14.1% win rate
  - Selective operator eliminated all puts (put model AUC=0.50, can't distinguish good/bad puts with matured features)
  - 65% of trades hit stop-loss in avg 1.9 days — stops too tight for options volatility
  - Only 1% hit take-profit — 50% TP too high for daily-checked options
- ✅ Rebuilt candidate dataset with EXIT-STRATEGY-REALIZED labels (not terminal returns).
  `good_trade` now = realized return > 0 under bucketed TP/SL/max-hold simulation.
  Call good_trade rate: 55.6% (up from 44.6%), put: 21.9%.
  Result: identical backtest ($3,634) — labels improved but meta-model still can't
  distinguish good from bad with matured-only efficacy features + 5-day delay.
### Honest assessment of where we stand (2026-04-01):
The ranker achieves NDCG@20 = 0.601 and the full causal stack produces -63.7% return.
The system is NOT profitable yet. Three paths remain:

1. **Widen stops dramatically** — options are too volatile for 15-25% daily stops.
   Try no stop-loss at all (pure 5-day hold was -85% vs -91% with stops).
   Or try very wide stops (50%+) that only trigger on true disasters.

2. **Better meta-model features** — the 5-day maturity delay kills efficacy signal.
   With hourly data (another agent is building this), intraday features could
   give the meta-model real-time signal without waiting for 5-day settlement.

3. **Bug 5: Relevance bins** — increase from 5 to 15-20 bins, or use continuous
   relevance, or switch to ApproxNDCG loss. This could push NDCG from 0.60 to 0.70+.

4. **Train on exit-strategy-realized returns** (the ranker itself, not just meta-models).
   Currently ranker trains on terminal 5-day return. If trained on bucketed exit returns,
   it would learn to pick options that hit TP before hitting SL.

5. **Use `p_good * score` to rank survivors** instead of just gating.

### Key model artifacts:
- `/tmp/neural_ranker_corrected.pt` — best ranker (NDCG 0.601, embed=256, corrected codebase)
- GCS `base_model_0634.pt` — old best (0.634, pre-correction, NOT on corrected universe)
- GCS `neural_ranker_artifact.pt` — latest (0.509, runner-up params, NOT the best)

## Python Environment

**Location:** `/Users/chinonsoisiodu/Documents/global_python_env/bin/python`
**Key packages:** torch 2.8, pandas, numpy, scikit-learn, xgboost, optuna, pyarrow, pyyaml

**Running locally:** Always set `PYTHONPATH=../updated_option_agent_codebase:.` because `utils.py` imports from the parent codebase.

## GCP Project

- **Project:** gen-lang-client-0478886850
- **Region:** us-east4
- **GPU quota:** 1 NVIDIA L4 at a time
- **Jobs:** `gpu-poc-test` (main training job), `optuna-sweep-1/2/3`, `target-test-a/b/c/d`

## Key Artifacts on GCS

- `gs://neural-ranker-training-data/artifacts/base_model_0634.pt` — best ranker (2018-2023 → 2024-2025, NDCG 0.634)
- `gs://neural-ranker-training-data/artifacts/neural_ranker_artifact.pt` — latest (may be overwritten by subsequent runs)
- `gs://neural-ranker-training-data/parquet/year_*_prepared.parquet` — pre-processed training data
