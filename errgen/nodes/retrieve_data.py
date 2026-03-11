"""
retrieve_data node

Collects raw financial data and news from external APIs:
  - FMP (Financial Modeling Prep): company profile, income statement,
    balance sheet, cash flow, stock news
  - NewsAPI: keyword-based supplemental news
  - Finnhub: company news (optional, only when FINNHUB_API_KEY is set)

All raw API responses are converted to EvidenceChunk + SourceMetadata
objects.  Returns the new items to be appended to the graph state via
the list-add reducer.
"""

from __future__ import annotations

import logging

from errgen.config import Config
from errgen.data import FMPClient, FinnhubClient, NewsAPIClient
from errgen.models import EvidenceChunk, SourceMetadata

logger = logging.getLogger(__name__)


def _date_range(as_of_date: str | None) -> tuple[str | None, str | None]:
    """Return (from_date, to_date) for news API queries based on as_of_date."""
    if not as_of_date:
        return None, None
    try:
        if len(as_of_date) == 7:          # "YYYY-MM"
            year, month = int(as_of_date[:4]), int(as_of_date[5:7])
        else:
            parts = as_of_date.split("-")
            year, month = int(parts[0]), int(parts[1])
        from_date = f"{year - 1}-{month:02d}-01"
        to_date = as_of_date if len(as_of_date) > 7 else f"{as_of_date}-28"
        return from_date, to_date
    except Exception:
        return None, None


def retrieve_data(state: dict) -> dict:
    """Fetch financial statements and news; return new sources and chunks."""
    ticker: str = state["ticker"]
    company_name: str = state.get("company_name") or ticker
    as_of: str | None = state.get("as_of_date")

    logger.info("retrieve_data: collecting data for %s (as_of=%s)", ticker, as_of)

    fmp = FMPClient()
    newsapi = NewsAPIClient()
    finnhub = FinnhubClient() if Config.FINNHUB_API_KEY else None

    from_date, to_date = _date_range(as_of)
    new_sources: list[SourceMetadata] = []
    new_chunks: list[EvidenceChunk] = []
    updates: dict = {}

    # -- Company profile (also infers company name if not already set) ----
    src, cks = fmp.get_company_profile(ticker)
    new_sources.append(src)
    new_chunks.extend(cks)

    if not company_name or company_name == ticker:
        for ck in cks:
            if ck.field_name == "overview":
                try:
                    inferred = ck.text.split("(")[0].strip()
                    if inferred:
                        updates["company_name"] = inferred
                        company_name = inferred
                except Exception:
                    pass
                break

    # -- Financial statements ---------------------------------------------
    for label, fetch in [
        ("income statement", lambda: fmp.get_income_statement(ticker, "annual")),
        ("balance sheet",    lambda: fmp.get_balance_sheet(ticker, "annual")),
        ("cash flow",        lambda: fmp.get_cash_flow(ticker, "annual")),
    ]:
        try:
            src, cks = fetch()
            new_sources.append(src)
            new_chunks.extend(cks)
            logger.info("retrieve_data: %s – %d chunks", label, len(cks))
        except Exception as exc:
            logger.warning("retrieve_data: %s failed: %s", label, exc)

    # -- FMP stock news ---------------------------------------------------
    try:
        src, cks = fmp.get_stock_news(
            ticker,
            limit=Config.MAX_NEWS_ARTICLES,
            from_date=from_date,
            to_date=to_date,
        )
        new_sources.append(src)
        new_chunks.extend(cks)
        logger.info("retrieve_data: FMP news – %d chunks", len(cks))
    except Exception as exc:
        logger.warning("retrieve_data: FMP news failed: %s", exc)

    # -- NewsAPI (supplemental) -------------------------------------------
    try:
        src, cks = newsapi.get_company_news(
            ticker=ticker,
            company_name=company_name,
            from_date=from_date,
            to_date=to_date,
            page_size=Config.MAX_NEWS_ARTICLES,
        )
        new_sources.append(src)
        new_chunks.extend(cks)
        logger.info("retrieve_data: NewsAPI – %d chunks", len(cks))
    except Exception as exc:
        logger.warning("retrieve_data: NewsAPI failed: %s", exc)

    # -- Finnhub (optional) -----------------------------------------------
    if finnhub:
        try:
            src, cks = finnhub.get_company_news(
                ticker=ticker,
                from_date=from_date,
                to_date=to_date,
                limit=Config.MAX_NEWS_ARTICLES,
            )
            new_sources.append(src)
            new_chunks.extend(cks)
            logger.info("retrieve_data: Finnhub – %d chunks", len(cks))
        except Exception as exc:
            logger.warning("retrieve_data: Finnhub failed: %s", exc)

    logger.info(
        "retrieve_data: complete – %d sources, %d chunks",
        len(new_sources), len(new_chunks),
    )

    updates["sources"] = new_sources
    updates["evidence_chunks"] = new_chunks
    return updates
