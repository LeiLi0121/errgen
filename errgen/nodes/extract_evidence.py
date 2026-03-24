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
from datetime import date, timedelta

from errgen.calculator.finance_calc import FinanceCalculator, build_growth_request
from errgen.config import Config
from errgen.extraction import FinancialExtractor, NewsExtractor
from errgen.models import (
    CalculationRequest,
    EvidenceChunk,
    ExtractedFact,
    SourceMetadata,
    SourceType,
)

logger = logging.getLogger(__name__)


def _price_rows(source: SourceMetadata) -> list[dict]:
    historical = source.metadata.get("historical") or []
    rows = [row for row in historical if isinstance(row, dict) and row.get("date") and row.get("close") is not None]
    return sorted(rows, key=lambda row: row["date"])


def _nearest_price(rows: list[dict], target_date: date) -> dict | None:
    chosen: dict | None = None
    for row in rows:
        try:
            row_date = date.fromisoformat(str(row["date"])[:10])
        except ValueError:
            continue
        if row_date <= target_date:
            chosen = row
        else:
            break
    return chosen


def _extract_price_signals(
    ticker: str,
    sources: list[SourceMetadata],
) -> tuple[list[ExtractedFact], list[CalculationRequest], list]:
    calc = FinanceCalculator()
    facts: list[ExtractedFact] = []
    calc_reqs: list[CalculationRequest] = []
    calc_results = []

    stock_source = next(
        (
            s for s in sources
            if s.source_type == SourceType.PRICE_DATA
            and s.metadata.get("series_role") == "stock"
        ),
        None,
    )
    benchmark_source = next(
        (
            s for s in sources
            if s.source_type == SourceType.PRICE_DATA
            and s.metadata.get("series_role") == "benchmark"
        ),
        None,
    )
    if not stock_source:
        return facts, calc_reqs, calc_results

    stock_rows = _price_rows(stock_source)
    benchmark_rows = _price_rows(benchmark_source) if benchmark_source else []
    if not stock_rows:
        return facts, calc_reqs, calc_results

    latest_stock = stock_rows[-1]
    latest_price = float(latest_stock["close"])
    facts.append(
        ExtractedFact(
            chunk_ids=[],
            fact_type="latest_price",
            subject=ticker,
            period=str(latest_stock["date"])[:10],
            value=latest_price,
            unit="USD",
            description=f"{ticker} latest close price: ${latest_price:.2f}",
        )
    )

    windows = [
        ("30d", Config.PREDICTION_LOOKBACK_DAYS),
        ("1y", min(Config.PRICE_LOOKBACK_DAYS, 365)),
    ]
    computed_returns: dict[str, float] = {}
    benchmark_returns: dict[str, float] = {}

    end_date = date.fromisoformat(str(latest_stock["date"])[:10])
    for label, days in windows:
        start_target = end_date - timedelta(days=days)
        stock_start = _nearest_price(stock_rows, start_target)
        if stock_start and float(stock_start["close"]) > 0:
            req = build_growth_request(
                description=f"{ticker} {label} price return",
                current=latest_price,
                previous=float(stock_start["close"]),
            )
            calc_reqs.append(req)
            result = calc.compute(req)
            calc_results.append(result)
            computed_returns[label] = float(result.result)

        if benchmark_rows:
            benchmark_end = benchmark_rows[-1]
            benchmark_start = _nearest_price(benchmark_rows, start_target)
            if benchmark_start and float(benchmark_start["close"]) > 0 and float(benchmark_end["close"]) > 0:
                req = build_growth_request(
                    description=f"{benchmark_source.ticker} {label} benchmark return",
                    current=float(benchmark_end["close"]),
                    previous=float(benchmark_start["close"]),
                )
                calc_reqs.append(req)
                result = calc.compute(req)
                calc_results.append(result)
                benchmark_returns[label] = float(result.result)

        if label in computed_returns and label in benchmark_returns:
            req = CalculationRequest(
                operation="difference",
                inputs={"a": computed_returns[label], "b": benchmark_returns[label]},
                description=f"{ticker} {label} excess return vs {benchmark_source.ticker}",
            )
            calc_reqs.append(req)
            calc_results.append(calc.compute(req))

    return facts, calc_reqs, calc_results


def extract_evidence(state: dict) -> dict:
    """Extract structured facts and run deterministic financial calculations."""
    ticker: str = state["ticker"]
    as_of: str | None = state.get("as_of_date")
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    sources: list[SourceMetadata] = state.get("sources") or []

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

    price_facts, price_calc_reqs, price_calc_results = _extract_price_signals(
        ticker=ticker,
        sources=sources,
    )
    facts.extend(price_facts)
    calc_reqs.extend(price_calc_reqs)
    calc_results.extend(price_calc_results)

    logger.info(
        "extract_evidence: %d facts, %d calc results for %s",
        len(facts), len(calc_results), ticker,
    )

    return {
        "extracted_facts": facts,
        "calculation_requests": calc_reqs,
        "calculations": calc_results,
    }
