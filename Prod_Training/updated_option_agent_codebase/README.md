# Updated options-agent research stack

This bundle now contains two layers:

- **P0/P1 baseline stack**: leakage-aware ranker, meta-labeler, return model, constrained allocator, and stateful backtest.
- **P2/P3 advanced stack**: chain-surface latent features, regime-aware experts, conformalized return intervals, and an agent watchdog / control plane.

## What is new in P2/P3

The advanced stack adds four concrete capabilities on top of the safer baseline:

1. **Date-local chain surface encoder**
   - Builds same-date option-chain summaries such as call/put skew, term slope, smile curvature, liquidity state, and latent PCA surface factors.
   - These features are fit only on the training partition and reused at inference.

2. **Regime-aware experts**
   - Learns date-level market regimes from observable state variables and surface features.
   - Blends a global model with regime-specific experts for both profitability and return forecasting.

3. **Conformalized return uncertainty**
   - The advanced return model outputs lower / mean / upper forecasts and widens intervals with calibration-derived conformal corrections.
   - This makes uncertainty estimates more conservative and operationally safer.

4. **Agent watchdog / safety layer**
   - Computes drift and health diagnostics.
   - Downgrades recommendations to `shadow_only` or `halt` when feature drift or prediction instability becomes too high.
   - Exposes deterministic decision packets for a GPT-5.4 agent to consume as tools.

## Files

### Baseline
- `utils.py` — safe data loading, schema normalization, feature engineering, target creation, purged splits.
- `prod_train_ranker.py` — XGBoost ranker with walk-forward OOF diagnostics.
- `prod_meta_labeler.py` — calibrated profitability model trained on honest ranker features.
- `prod_log_return_predictor.py` — distributional return model trained on the same target definition.
- `prod_hybrid_kelly.py` — prediction service, constrained allocator, stateful backtest.

### P2/P3
- `advanced_utils.py` — chain-surface encoder, regime router, and drift watchdog.
- `prod_advanced_stack.py` — advanced training and inference pipeline.
- `prod_agent_watchdog.py` — deterministic tool layer for an LLM agent.

### Tests
- `tests/test_leakage_and_smoke.py` — baseline leakage and smoke tests.
- `tests/test_advanced_stack.py` — P2/P3 smoke tests and watchdog checks.

## Recommended workflow

### Baseline artifacts
```bash
python prod_train_ranker.py --data /path/to/year_2025_data.csv --config ./config.yaml --output-dir ./model_output
python prod_meta_labeler.py --data /path/to/year_2025_data.csv --config ./config.yaml --output-dir ./model_output --ranker-artifact ./model_output/ranker_artifact.joblib
python prod_log_return_predictor.py --data /path/to/year_2025_data.csv --config ./config.yaml --output-dir ./model_output --ranker-artifact ./model_output/ranker_artifact.joblib
python prod_hybrid_kelly.py --data /path/to/year_2025_data.csv --config ./config.yaml --output-dir ./model_output --ranker-artifact ./model_output/ranker_artifact.joblib --meta-artifact ./model_output/meta_labeler_artifact.joblib --return-artifact ./model_output/return_distribution_artifact.joblib
```

### Advanced P2/P3 artifact
```bash
python prod_advanced_stack.py train --data /path/to/year_2025_data.csv --config ./config.yaml --output-dir ./advanced_output
python prod_advanced_stack.py run --data /path/to/new_option_data.csv --config ./config.yaml --advanced-artifact ./advanced_output/advanced_trade_stack_artifact.joblib --output-dir ./advanced_scoring
```

### Agent-safe decision packet
```bash
python prod_agent_watchdog.py --advanced-artifact ./advanced_output/advanced_trade_stack_artifact.joblib --data /path/to/new_option_data.csv --config ./config.yaml --output-dir ./agent_packets
```

That emits one of three execution modes:
- `proceed`
- `shadow_only`
- `halt`

## Notes on safety and leakage

- Feature engineering remains point-in-time safe.
- The advanced stack trains its saved artifact on the historical train partition and evaluates on the post-purge holdout partition.
- Regime-aware expert models never use calibration targets during fitting.
- The watchdog is deterministic and does not place trades.

## Important caveat

This is a materially stronger research and paper-trading stack than the original prototype, but it still does **not** guarantee live profitability or the absence of every possible bug. Keep it in paper / shadow mode first, especially when the watchdog returns `shadow_only` or `halt`.
