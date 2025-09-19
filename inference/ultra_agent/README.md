# Ultra-Agent Stress Engine (Prototype)

This module houses the next-generation stress-testing agent that sits on top of
our deterministic CQF + Monte Carlo stack. The agent’s job is to propose,
score, and curate a compact pack of high-value scenarios that expose the
portfolio’s weak spots under current regimes.

Key ideas:

- **LLM-assisted scenario generation** with strict guardrails.
- **Oracle scoring** using the existing `prod_stress_mc` logic.
- **Plausibility checks** (copula likelihood, IV sanity, correlation caps).
- **Diversity optimisation** so the pack spans distinct risk angles.
- **Adversarial search** within bounded shock space.
- **Deterministic scenario packs** ready for replay in production.

This directory currently contains scaffolding for the agent. Individual modules
are documented inline. Nothing here alters the production pipeline yet—the
integration will happen once the agent is assembled and validated.
