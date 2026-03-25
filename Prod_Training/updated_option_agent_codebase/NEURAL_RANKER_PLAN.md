# Neural Ranker Plan: NDCG@20 > 0.90

## Current State

- XGBoost ranker achieves NDCG@20 = 0.723 after Optuna tuning on 2023-2024 data
- Hyperparameter search has converged — further tuning yields diminishing returns
- XGBoost ranks options independently (pointwise), missing cross-option structure

## Why a Neural Ranker

XGBoost treats each option as an independent row. It cannot learn patterns like:
- "This call is cheap relative to adjacent strikes" (cross-strike signal)
- "IV skew is steeper than usual for this tenor" (surface-relative signal)
- "The top-ranked options share concentrated gamma exposure" (portfolio-aware ranking)

A listwise neural ranker sees the entire daily option chain as one input and learns to rank the full set jointly. This is the difference between scoring items in isolation vs learning relative ordering.

## Proposed Architecture

### Option Chain Transformer

```
Daily chain (~9K options)
    |
[Per-option MLP encoder] -> 64-128 dim embeddings
    |
[2-3 Transformer layers with self-attention]
    |
    Each option attends to all others in the chain
    Learns cross-strike, cross-tenor, cross-type relationships
    |
[Ranking head] -> scalar score per option
    |
ListMLE or LambdaRank loss (listwise ranking loss)
```

### Model Size

- Embedding dim: 64-128
- Transformer layers: 2-3
- Heads: 4-8
- Total parameters: ~500K-2M
- This is tiny — comparable to a small image classifier

### Input Features Per Option

Current features (48 numerical + type):
- Greeks: delta, gamma, theta, vega, rho, IV
- Price: bid, ask, mid, last, strike
- Liquidity: volume, OI, spread, liquidity_score
- Market: SPY close, SMA50, RSI, MACD, VIX
- Engineered: moneyness, time_value_ratio, dist_to_atm, rolling stats, cross-sectional ranks

Additional features from P2/P3 surface encoder:
- surface_iv_mean, surface_term_slope, surface_smile_curvature
- surface_call_put_spread, surface_volume_per_contract
- PCA latent surface factors
- regime_id, regime_confidence

### Why This Should Work

1. Self-attention lets each option "see" the full chain — learns relative value
2. Positional encoding by (moneyness, DTE) gives the model surface structure
3. ListMLE loss directly optimizes the ranking metric, not a proxy
4. Surface features from P2/P3 are already computed — just add them as inputs

## Training Plan

### Hardware

- Apple M5 Mac, 10 cores, 24GB RAM
- PyTorch 2.8 with MPS (Metal GPU) confirmed working
- Training on MPS: expected ~5-10 min per epoch for 500 dates x 9K options

### Data

- Train: 2019-2023 (~1,250 dates)
- Validate: 2024 (252 dates)
- Test: 2025 (44 dates, never seen)
- Each "sample" is one full day's option chain
- DataLoader collates variable-length chains with padding + attention masks

### Loss Function

**ListMLE** (listwise ranking loss) or **LambdaRank**:
- ListMLE: likelihood of the observed permutation under the model's scores
- LambdaRank: pairwise loss weighted by NDCG delta — directly optimizes NDCG
- Both are differentiable and well-supported in PyTorch

### Training Loop

```
for epoch in range(50):
    for date_chain in train_loader:
        options = date_chain["features"]       # (N_options, D_features)
        relevance = date_chain["relevance"]    # (N_options,)

        scores = model(options)                # (N_options,)
        loss = lambda_rank_loss(scores, relevance)

        loss.backward()
        optimizer.step()

    val_ndcg = evaluate(model, val_loader)     # NDCG@20 on 2024
```

### Walk-Forward Safety

- Same purged walk-forward split logic as XGBoost pipeline
- No future data leakage — train on dates < purge cutoff
- Surface features refit per fold (same as P2/P3 advanced stack)

## Integration with Existing Pipeline

The neural ranker replaces only Stage 1 (option selection). The rest stays:

```
[Neural Ranker]  ->  scores + ranks + percentiles
       |
[Meta-Labeler]   ->  P(profit) calibrated
       |
[Return Model]   ->  quantile return forecasts
       |
[Hybrid Kelly]   ->  position sizing + backtest
```

The ranker output format (ranker_score, ranker_rank, ranker_percentile) is identical — downstream models don't need to change.

## Alternatives to Explore

1. **DeepSets** — Simpler than Transformer, permutation-invariant aggregation. Good baseline.
2. **SetTransformer** — Inducing-point attention for efficiency with 9K items.
3. **Cross-attention with surface summary** — Encode the IV surface separately, then cross-attend.
4. **Mixture of Experts by regime** — Route to regime-specific ranking heads (extends P2/P3 regime router).

## Expected Outcome

- NDCG@20: 0.82-0.90+ (from 0.72 with XGBoost)
- Better top-5 selection quality (more profitable trades surfaced)
- The meta-labeler and return model should also improve since they consume ranker features

## Implementation Order

1. Build a minimal ListMLE transformer ranker (single file, ~300 lines)
2. Train on 2023-2024 with MPS, validate NDCG lift over XGBoost
3. If NDCG > 0.80, integrate into the pipeline as an alternative ranker
4. A/B test XGBoost vs neural ranker on 2025 holdout
5. If confirmed, make it the default and retune meta/return models on the new ranker features
