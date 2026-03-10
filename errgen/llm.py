"""
OpenAI LLM client wrapper.

Provides:
  - chat()      – standard text completion
  - chat_json() – JSON-mode completion (response must be valid JSON)

Both methods include exponential-backoff retry on transient failures.
All calls are logged at DEBUG level with token usage.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

from errgen.config import Config

logger = logging.getLogger(__name__)

# Module-level client (lazy-initialised so import doesn't fail without key)
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request and return the assistant message content.

    Retries on rate-limit and transient API errors with exponential backoff.
    """
    model = model or Config.OPENAI_MODEL
    temperature = temperature if temperature is not None else Config.OPENAI_TEMPERATURE
    max_tokens = max_tokens or Config.OPENAI_MAX_TOKENS

    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(Config.LLM_RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            logger.debug(
                "LLM call ok | model=%s | prompt_tokens=%s | completion_tokens=%s",
                model,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
            )
            return content.strip()

        except RateLimitError as exc:
            wait = Config.LLM_RETRY_DELAY * (2 ** attempt)
            logger.warning(
                "OpenAI rate limit hit (attempt %d/%d). Waiting %.1fs. %s",
                attempt + 1,
                Config.LLM_RETRY_ATTEMPTS,
                wait,
                exc,
            )
            last_exc = exc
            time.sleep(wait)

        except (APIError, APIConnectionError) as exc:
            wait = Config.LLM_RETRY_DELAY * (2 ** attempt)
            logger.warning(
                "OpenAI API error (attempt %d/%d). Waiting %.1fs. %s",
                attempt + 1,
                Config.LLM_RETRY_ATTEMPTS,
                wait,
                exc,
            )
            last_exc = exc
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI call failed after {Config.LLM_RETRY_ATTEMPTS} attempts. "
        f"Last error: {last_exc}"
    )


def chat_json(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """
    Like chat() but sets response_format=json_object and parses the result.

    Raises ValueError if the response is not valid JSON.
    The caller is responsible for validating the parsed structure.
    """
    model = model or Config.OPENAI_MODEL
    temperature = temperature if temperature is not None else Config.OPENAI_TEMPERATURE
    max_tokens = max_tokens or Config.OPENAI_MAX_TOKENS

    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(Config.LLM_RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            usage = resp.usage
            logger.debug(
                "LLM JSON call ok | model=%s | prompt_tokens=%s | completion_tokens=%s",
                model,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
            )
            try:
                return json.loads(content)
            except json.JSONDecodeError as parse_err:
                raise ValueError(
                    f"LLM returned non-JSON content: {content[:200]}"
                ) from parse_err

        except RateLimitError as exc:
            wait = Config.LLM_RETRY_DELAY * (2 ** attempt)
            logger.warning(
                "OpenAI rate limit hit (JSON, attempt %d/%d). Waiting %.1fs.",
                attempt + 1,
                Config.LLM_RETRY_ATTEMPTS,
                wait,
            )
            last_exc = exc
            time.sleep(wait)

        except (APIError, APIConnectionError) as exc:
            wait = Config.LLM_RETRY_DELAY * (2 ** attempt)
            logger.warning(
                "OpenAI API error (JSON, attempt %d/%d). Waiting %.1fs.",
                attempt + 1,
                Config.LLM_RETRY_ATTEMPTS,
                wait,
            )
            last_exc = exc
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI JSON call failed after {Config.LLM_RETRY_ATTEMPTS} attempts. "
        f"Last error: {last_exc}"
    )


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    """Convenience helper to build a two-message conversation list."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
