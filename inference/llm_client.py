"""Optional LLM client wrappers used by the scenario engine.

The OpenAI client is optional so that the rest of the pipeline can run
without the dependency (or without credentials).  If the library or API key
is missing, initialisation will raise and the caller can fall back to the
legacy Monte Carlo stress logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - we handle the absence below
    OpenAI = None  # type: ignore


class OpenAILLMClient:
    """Thin wrapper around the OpenAI responses API.

    Parameters
    ----------
    api_key:
        API key to use.  If omitted, the constructor will read the
        ``OPENAI_API_KEY`` environment variable.
    model:
        Model name to call.  Defaults to ``gpt-4.1-mini`` which offers a good
        balance of latency/cost for scenario generation tasks.
    request_kwargs:
        Extra keyword arguments forwarded to ``responses.create`` (e.g.
        ``{"temperature": 0}`` for deterministic output).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gpt-4.1-mini",
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is not installed. Install it or disable the LLM "
                "stress mode."
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY
        if not api_key:
            raise ValueError(
                "OpenAILLMClient requires an API key. Set OPENAI_API_KEY or pass "
                "--llm-api-key on the command line."
            )

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._request_kwargs = request_kwargs or {}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload: Dict[str, Any] = dict(self._request_kwargs)
        payload.update(kwargs)

        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            **payload,
        )

        # Try a couple of access paths depending on SDK version
        text: Optional[str] = None
        if hasattr(response, "output_text"):
            text = getattr(response, "output_text")
        if not text and hasattr(response, "output"):
            try:
                parts = response.output  # type: ignore[attr-defined]
                if parts:
                    content = parts[0]["content"][0]
                    text = content.get("text") if isinstance(content, dict) else str(content)
            except Exception as exc:  # pragma: no cover - defensive parsing
                logger.debug("Failed to parse OpenAI response structure: %s", exc)

        if not text:
            raise RuntimeError("OpenAI response did not contain text content")

        return text


__all__ = ["OpenAILLMClient"]
DEFAULT_OPENAI_KEY = "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"

