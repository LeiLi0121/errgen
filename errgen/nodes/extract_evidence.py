"""
extract_evidence node

Runs two extraction pipelines over the collected evidence chunks:

1. FinancialExtractor (rule-based, deterministic)
   Indexes numeric chunks and builds standard CalculationRequest objects.
   All arithmetic is executed by FinanceCalculator – never the LLM.

2. NewsExtractor (LLM-based)
   Calls the fast LLM to extract structured events/facts from news chunks.

Returns new extracted_facts, calculation_requests, and calculations to be
appended to the graph state.
"""

from __future__ import annotations

import logging

from errgen.extraction import FinancialExtractor, NewsExtractor
from errgen.models import EvidenceChunk, SourceType

logger = logging.getLogger(__name__)


def extract_evidence(state: dict) -> dict:
    """Extract structured facts and run deterministic financial calculations."""
    ticker: str = state["ticker"]
    as_of: str | None = state.get("as_of_date")
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]

    logger.info(
        "extract_evidence: processing %d chunks for %s", len(all_chunks), ticker
    )

    fin_extractor = FinancialExtractor()
    news_extractor = NewsExtractor()

    # 1. Deterministic financial extraction + calculations (no LLM arithmetic)
    facts, calc_reqs, calc_results = fin_extractor.extract(ticker, all_chunks)

    # 2. LLM-based news event extraction
    news_chunks = [c for c in all_chunks if c.source_type == SourceType.NEWS]
    if news_chunks:
        news_facts = news_extractor.extract(
            ticker=ticker,
            news_chunks=news_chunks,
            as_of_date=as_of,
        )
        facts.extend(news_facts)

    logger.info(
        "extract_evidence: %d facts, %d calc results for %s",
        len(facts), len(calc_results), ticker,
    )

    return {
        "extracted_facts": facts,
        "calculation_requests": calc_reqs,
        "calculations": calc_results,
    }
