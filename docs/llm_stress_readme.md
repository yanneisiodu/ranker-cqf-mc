# LLM Stress Engine Integration Notes

## Runtime switches
- `--stress-mode {mc,llm,shadow}` (default `mc`)
  - `mc`: legacy Monte Carlo output only (LLM pipeline skipped).
  - `llm`: deterministic LLM-driven stress results replace the legacy output and are written to `stress_metrics.csv`. The MC results are still computed for logging.
  - `shadow`: legacy MC remains authoritative, while LLM metrics are emitted to `stress_metrics_llm.csv` for comparison.
- `--llm-provider` / `--llm-model` / `--llm-api-key`: choose the backend (`openai` today), override the model identifier, and optionally supply an API key. **Prefer setting `OPENAI_API_KEY` in your environment instead of committing keys to source control.**
- `--llm-log-scenarios`: include per-underlying scenario payloads in the `_scenario_log` column of the LLM results (disabled by default to keep files light-weight).
- `--llm-max-contracts`: cap how many contracts are considered when LLM mode is active (default 20). If `--top-n` is larger, it is automatically reduced and logged.
- `--llm-max-groups`: hard limit on the number of LLM prompts (default 10). If grouping underlyings would exceed this cap, the run falls back to the Monte Carlo stress output.

## Artifacts
- Always written: `stress_metrics.csv`, `trade_recommendations.csv`.
- When `--stress-mode` in `{llm, shadow}`: additional `stress_metrics_llm.csv` (sorted by utility) plus optional scenario logs when `--llm-log-scenarios` is set.

## Configuration
- Scenario generation uses `ScenarioConfig` (defaults in code) allowing:
  - `group_key` (`underlying` by default, auto-fallbacks to `contractID` when absent),
  - scenario bounds (`spot_return_bounds`, `vol_change_bounds`), minimum probability, and fallback shock templates.
- Numerical layer uses `LLMStressConfig`, mirroring the legacy Monte Carlo tuning:
  - `risk_aversion`, `min_prob_profit`, `max_downside_var`, `skew_bonus`, and optional calibration offsets (`prob_calibrator`, `var_adjust`, `cvar_adjust`).
- Deterministic prompts: when the OpenAI client is instantiated (temperature forced to 0) outputs are reproducible for a given prompt.

## Logging & auditing
- All LLM results are written via CSV; additional structured logs can be captured from `_scenario_log` when enabled.
- Fallback handling (validation failure, missing response) is visible via standard logging messages—look for `Saved LLM stress metrics` confirmation and warnings in the pipeline log.
- The shadow mode provides a direct comparison channel without affecting downstream ranking/filters.

## Next steps
- Wire a real `LLMClient` implementation to `ScenarioEngine` once credentials and prompts are finalised.
- Plug in calibrated assets (probability calibrator, VaR/CVaR offsets) via `LLMStressConfig` when available.
- Extend validation to incorporate retrieval-based analogues and richer narratives as described in the design doc.
