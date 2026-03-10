"""
News extractor.

Uses an LLM to distil key events, themes, and sentiment signals from raw
news evidence chunks into structured ExtractedFact objects.

Each extracted fact carries the chunk_ids of the articles that support it,
ensuring that every downstream analysis paragraph can trace back to the
original news source.
"""

from __future__ import annotations

import json
import logging

from errgen.config import Config
from errgen.llm import build_messages, chat_json
from errgen.models import EvidenceChunk, ExtractedFact, SourceType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert financial news analyst.
You will receive a batch of news article chunks about a company.
Your task is to extract the most important facts, events, and signals.

Return a JSON object with the following structure:
{
  "facts": [
    {
      "fact_type": "one of: product_launch | partnership | financial_result | regulatory | management_change | market_trend | geopolitical | analyst_rating | competitive | other",
      "description": "A concise 1–3 sentence factual description of the event or signal. Be specific.",
      "period": "approximate date or period if known (e.g. '2024-Q4' or '2024-11')",
      "sentiment": "positive | negative | neutral | mixed",
      "importance": "high | medium | low",
      "supporting_chunk_ids": ["id1", "id2"]
    }
  ]
}

Rules:
- Only extract facts clearly supported by the provided chunks.
- Include the chunk IDs of the articles that support each fact.
- Do NOT invent information not present in the chunks.
- Be specific: include company names, dates, numbers when available.
- Limit to the most relevant 15 facts maximum.
- If a fact appears in multiple articles, cite all relevant chunk IDs.
"""


class NewsExtractor:
    """
    Extracts structured news facts from raw EvidenceChunk objects of type NEWS.

    Uses GPT-4o-mini by default (cheaper, sufficient for extraction).
    """

    def extract(
        self,
        ticker: str,
        news_chunks: list[EvidenceChunk],
        as_of_date: str | None = None,
    ) -> list[ExtractedFact]:
        """
        Extract structured facts from a list of news evidence chunks.

        Returns a list of ExtractedFact objects, each referencing the
        chunk_ids that support them.
        """
        if not news_chunks:
            logger.warning("NewsExtractor: no news chunks provided for %s", ticker)
            return []

        # Build chunk reference block for the LLM
        chunk_lines: list[str] = []
        for chunk in news_chunks:
            chunk_lines.append(
                f"[CHUNK_ID: {chunk.chunk_id}]\n{chunk.text[:800]}\n---"
            )

        user_content = (
            f"Company/Ticker: {ticker}\n"
            + (f"As-of date constraint: {as_of_date}\n" if as_of_date else "")
            + f"\nNews chunks ({len(news_chunks)} articles):\n\n"
            + "\n".join(chunk_lines)
        )

        messages = build_messages(_SYSTEM_PROMPT, user_content)

        try:
            raw = chat_json(messages, model=Config.OPENAI_FAST_MODEL)
        except Exception as exc:
            logger.error("NewsExtractor LLM call failed for %s: %s", ticker, exc)
            return []

        facts_raw: list[dict] = raw.get("facts", [])
        facts: list[ExtractedFact] = []

        # Map chunk_id → chunk for validation
        chunk_id_set = {c.chunk_id for c in news_chunks}

        for item in facts_raw:
            # Validate that cited chunk IDs actually exist
            cited_ids = [
                cid for cid in item.get("supporting_chunk_ids", [])
                if cid in chunk_id_set
            ]
            if not cited_ids:
                # The LLM invented a chunk ID; skip this fact
                logger.debug(
                    "NewsExtractor: dropping fact with no valid chunk IDs: %s",
                    item.get("description", ""),
                )
                continue

            facts.append(
                ExtractedFact(
                    chunk_ids=cited_ids,
                    fact_type=item.get("fact_type", "other"),
                    subject=ticker,
                    period=item.get("period"),
                    description=item.get("description", ""),
                    confidence=0.9,
                )
            )

        logger.info(
            "NewsExtractor: extracted %d facts from %d chunks for %s",
            len(facts), len(news_chunks), ticker,
        )
        return facts
