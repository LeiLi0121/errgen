"""
parse_query node

Converts a natural-language equity research request into structured fields:
  ticker, company_name, focus areas, as_of_date.

This is the graph entry point and lets callers pass pure natural language
without pre-parsing anything.
"""

from __future__ import annotations

import logging
import uuid

from errgen.config import Config
from errgen.llm import build_messages, chat_json

logger = logging.getLogger(__name__)

_SYSTEM = """You are a financial research assistant.
Extract structured information from a natural-language equity research request.

Return ONLY valid JSON with these exact fields:
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "focus": ["AI chips", "financial analysis", "risks"],
  "as_of_date": "2025-01-31"
}

Rules:
- ticker must be UPPERCASE (e.g. NVDA, AAPL, MSFT, TSLA, AMZN)
- company_name: full official company name
- focus: 3–6 relevant analysis topics inferred from the request
- as_of_date:
  - use "YYYY-MM-DD" when the request gives an exact date
  - use "YYYY-MM" when only a month/period is given
  - otherwise null
"""


def parse_query(state: dict) -> dict:
    """Convert natural-language query into structured ticker / focus / date fields."""
    query: str = state["query"]
    logger.info("parse_query: parsing → '%s'", query[:100])

    messages = build_messages(_SYSTEM, f"Request: {query}")
    try:
        parsed = chat_json(messages, model=Config.OPENAI_FAST_MODEL)
    except Exception as exc:
        logger.error("parse_query: LLM call failed: %s", exc)
        raise RuntimeError(f"parse_query failed to parse query: {exc}") from exc

    ticker = (parsed.get("ticker") or "").strip().upper()
    if not ticker:
        raise ValueError(
            f"parse_query: could not extract a ticker from: '{query}'. "
            "Please include the company name or ticker symbol in your request."
        )

    company_name = (parsed.get("company_name") or "").strip()
    focus: list[str] = parsed.get("focus") or []
    as_of_date: str | None = parsed.get("as_of_date") or None
    run_id = str(uuid.uuid4())

    logger.info(
        "parse_query: ticker=%s company='%s' focus=%s as_of=%s run_id=%s",
        ticker, company_name, focus, as_of_date, run_id,
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "focus": focus,
        "as_of_date": as_of_date,
        "run_id": run_id,
    }
