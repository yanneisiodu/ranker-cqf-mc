# LLM-Augmented Stress Testing Implementation Plan

## 1. Purpose & Scope
- **Objective:** Embed an LLM scenario/orchestration layer around the existing `EnhancedStressMC` so stress testing becomes adaptive and explainable without giving up quantitative rigor.
- **In Scope:** Scenario generation, regime tagging, configuration tuning, logging/audit enhancements, safety checks, observability, and rollout.
- **Out of Scope (for now):** Replacing the MC math, changing CQF training, production execution changes.

## 2. Current State Snapshot
- Ranker → CQF → `EnhancedStressMC` → recommendations pipeline running daily, emitting CSV artifacts.
- Stress MC already supports deterministic sampling, risk gating (`prob_profit`, `VaR`, `CVaR`), scenario mixes, and parallel execution.
- No LLM components in production; only research notes and the legacy architecture blueprint.

## 3. Guiding Principles
1. **Quant stays in charge:** LLMs suggest scenarios and parameters; Monte Carlo computes risk metrics.
2. **Full reproducibility:** Every LLM decision is logged with prompt, response, UUID, and fallback path.
3. **Safe rollout:** Shadow → blended → full control, with measurable lift before each promotion.
4. **Human-in-the-loop:** Daily dashboard surfaces LLM narratives and MC outputs for analyst review.

## 4. Phase Plan (Technical Workstream)

### Phase 0 – Foundations (Week 0)
- Finalize personas (Strategist, Risk Analyst) and prompt templates.
- Stand up secrets/config management for LLM provider or local model.
- Define JSON schema for scenarios, regime labels, and config overrides.
- Document baseline fallback scenario set (legacy MC shocks) and per-underlying scenario caching contract.
- Capture calibration assets to load at runtime (probability calibrator, VaR/CVaR offsets) and logging requirements for deterministic prompts.
- Deliverable: `docs/llm_interface_spec.md`, staging access to chosen LLM.

**Design Doc Alignment Touchpoints**
- Scenario schema must cover `underlying_return_pct`, `vol_change`, `scenario_prob`, and narrative strings exactly as specified in the replacement design.
- Ensure deterministic prompting (temperature 0) and full prompt/response logging are part of the foundation deliverable.
- Define validation rules up front (shock clipping, no-arbitrage checks) so later phases can reuse them.
- Agree on weighted percentile implementation for VaR/CVaR and outline test cases (including conformal recalibration hooks).

### Phase 1 – Observation Harness (Weeks 1–2)
**Goal:** Run the LLM in read-only mode, logging suggested scenarios alongside current MC inputs.
- [ ] Build data summarizer module (CQF stats, regime features, SPY shocks, macro headlines).
- [ ] Implement `llm_scenario_suggester.py` returning structured JSON (scenarios, regime tag, rationale).
- [ ] Add comparison logger storing both heuristic and LLM suggestions (Parquet/CSV + markdown digest).
- [ ] Set up evaluation notebook to analyze differences across 7–10 trading days.
- Deliverables: daily `llm_scenario_shadow.jsonl`, comparison report, go/no-go criteria for Phase 2.

### Phase 2 – Scenario Assistant (Weeks 3–4)
**Goal:** Feed validated LLM scenarios into Stress MC in shadow mode.
- [ ] Implement schema validator + clipping (spot multiplier bounds, iv shock limits, probability normalization).
- [ ] Add per-underlying scenario caching layer and reuse checks to avoid redundant LLM calls.
- [ ] Extend `run_inference.py` to compute stress metrics twice (heuristic and LLM-driven) without affecting recommendations.
- [ ] Wire fallback logic: if LLM output fails validation, automatically substitute legacy MC scenarios and flag the run.
- [ ] Add metrics diff report (coverage, VaR/CVaR shifts, utility deltas, prob_profit Brier score) and alert thresholds.
- [ ] Introduce narrative logger (LLM rationale + resulting MC metrics) for analyst review.
- Exit Criteria: No schema failures for 10 consecutive runs; metrics within predefined guardrails and fallback rate < 5%.

### Phase 3 – Hybrid Control (Weeks 5–8)
**Goal:** Let the LLM influence production configs while preserving guardrails.
- [ ] Implement blending rules (e.g., weighted average scenarios, regime-specific overrides, fallbacks).
- [ ] Add safety checks: conformal coverage monitor, VaR monotonicity, crash-stats sanity.
- [ ] Build dashboard (Streamlit/Grafana) showing daily LLM narratives, scenario weights, MC outputs, and backtests.
- [ ] Run A/B or rolling-window backtest to compare live vs heuristic gating; require predefined uplift.
- Deliverables: `config/llm_mc_rules.yaml`, dashboard URL, rollout plan with alerting.

### Phase 4 – Production Hardening (Weeks 9–12)
**Goal:** Operationalize the hybrid stack with reliability, cost, and compliance controls.
- [ ] Move LLM inference to auto-scaling service (batching, cache, retry logic, fallback prompts).
- [ ] Implement audit trail writer (signed logs, retention policy, ticketing integration).
- [ ] Wire drift detection (prompt/response anomalies, scenario distribution drift, latency SLOs).
- [ ] Update runbooks, on-call procedures, and risk/compliance documentation.
- Exit Criteria: Two-week pilot with zero Sev-1 incidents, compliance sign-off, cost within budget target.

## 5. Validation & Metrics
- **Accuracy:** Compare VaR/CVaR coverage vs historical targets (<= 2pp deviation) using weighted percentiles; track prob_profit Brier score and utility improvements.
- **Stability:** Scenario parameter drift < 10% day-on-day unless regime change flagged; LLM rejection rate < 5%.
- **Explainability:** 100% of runs produce human-readable narratives linked to MC metrics.
- **Operational:** LLM inference P95 latency < 3 s; fallback activation < 1 per week.

## 6. Risks & Mitigations
| Risk | Mitigation |
| --- | --- |
| Hallucinated scenarios | JSON schema validation, clipping, fallback to heuristic set |
| Latency spikes | Caching, batching, async pre-fetch before MC stage |
| Cost overruns | Budget alerts, switch to fine-tuned local model if API cost > threshold |
| Analyst mistrust | Daily narrative dashboard, feedback loop, manual override control |
| Compliance concerns | Full logging, change management tickets, model risk review before Phase 3 |

## 7. Resource & Dependency Snapshot
- **Roles:** ML engineer (MC integration), LLM engineer (prompting + infra), Quant analyst (validation), DevOps/SRE (deployment), Product/Compliance stakeholders.
- **Dependencies:** LLM provider access, news data feed, existing CQF outputs, monitoring stack (Grafana/Prometheus), storage for logs.

## 8. Timeline Overview
| Phase | Duration | Target Window |
| --- | --- | --- |
| 0 | 1 week | Week 0 |
| 1 | 2 weeks | Weeks 1–2 |
| 2 | 2 weeks | Weeks 3–4 |
| 3 | 4 weeks | Weeks 5–8 |
| 4 | 4 weeks | Weeks 9–12 |

_Total effort: ~12 weeks assuming part-time allocation across two engineers; adjust for parallel staffing if available._

## 9. Next Immediate Actions
1. Confirm LLM provider (API vs self-host) and provision credentials.
2. Draft initial prompt templates for Strategist/Risk Analyst personas.
3. Review the `LLM-Based Replacement for Monte Carlo Stress Module` design doc and map its scenario fields/fallback rules into the interface spec.
4. Start Phase 0 tasks (interface spec + staging harness).
5. Schedule weekly sync with quant + product stakeholders for review checkpoints.

