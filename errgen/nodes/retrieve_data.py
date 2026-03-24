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
from datetime import date, timedelta

from errgen.config import Config
from errgen.data import FMPClient, FinnhubClient, NewsAPIClient, SECClient
from errgen.models import EvidenceChunk, SourceMetadata

logger = logging.getLogger(__name__)


def _source_has_history(source: SourceMetadata | None) -> bool:
    return bool(source and source.metadata.get("historical"))


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


def _lookback_window(as_of_date: str | None, days: int) -> tuple[str, str]:
    """Return YYYY-MM-DD window ending at as_of_date (or today if not set)."""
    if as_of_date:
        try:
            if len(as_of_date) == 7:
                end = date.fromisoformat(f"{as_of_date}-28")
            else:
                end = date.fromisoformat(as_of_date[:10])
        except ValueError:
            end = date.today()
    else:
        end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def retrieve_data(state: dict) -> dict:
    """Fetch financial statements and news; return new sources and chunks."""
    ticker: str = state["ticker"]
    company_name: str = state.get("company_name") or ticker
    as_of: str | None = state.get("as_of_date")

    logger.info("retrieve_data: collecting data for %s (as_of=%s)", ticker, as_of)

    fmp = FMPClient()
    newsapi = NewsAPIClient()
    finnhub = FinnhubClient() if Config.FINNHUB_API_KEY else None
    sec = SECClient()

    from_date, to_date = _date_range(as_of)
    filing_from, filing_to = _lookback_window(as_of, days=540)
    price_from, price_to = _lookback_window(as_of, days=Config.PRICE_LOOKBACK_DAYS)
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
        ("income statement", lambda: fmp.get_income_statement(ticker, "quarter")),
        ("balance sheet",    lambda: fmp.get_balance_sheet(ticker, "quarter")),
        ("cash flow",        lambda: fmp.get_cash_flow(ticker, "quarter")),
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
    # Finnhub company-news API requires both "from" and "to" (YYYY-MM-DD).
    # When as_of_date is not set we use a default range (past year → today).
    if finnhub:
        try:
            fh_from, fh_to = from_date, to_date
            if fh_from is None and fh_to is None:
                end = date.today()
                fh_from = (end - timedelta(days=365)).strftime("%Y-%m-%d")
                fh_to = end.strftime("%Y-%m-%d")
            src, cks = finnhub.get_company_news(
                ticker=ticker,
                from_date=fh_from,
                to_date=fh_to,
                limit=Config.MAX_NEWS_ARTICLES,
            )
            new_sources.append(src)
            new_chunks.extend(cks)
            logger.info("retrieve_data: Finnhub – %d chunks", len(cks))
        except Exception as exc:
            logger.warning("retrieve_data: Finnhub failed: %s", exc)

    # -- SEC filings (10-K / 10-Q) ---------------------------------------
    try:
        src, cks = sec.get_sec_filings(
            ticker=ticker,
            from_date=filing_from,
            to_date=filing_to,
            limit=Config.MAX_SEC_FILINGS,
        )
        new_sources.append(src)
        new_chunks.extend(cks)
        logger.info("retrieve_data: SEC filings – %d chunks", len(cks))
    except Exception as exc:
        logger.warning("retrieve_data: SEC filings failed: %s", exc)

    # -- Price history: company + benchmark -------------------------------
    for symbol, role in [
        (ticker, "stock"),
        (Config.BENCHMARK_TICKER, "benchmark"),
    ]:
        src: SourceMetadata | None = None
        cks: list[EvidenceChunk] = []
        try:
            src, cks = fmp.get_price_history(
                ticker=symbol,
                from_date=price_from,
                to_date=price_to,
            )
        except Exception as exc:
            logger.warning("retrieve_data: FMP %s price history failed: %s", role, exc)

        if not _source_has_history(src) and finnhub:
            try:
                logger.info(
                    "retrieve_data: FMP %s price history unavailable, falling back to Finnhub",
                    role,
                )
                src, cks = finnhub.get_price_history(
                    ticker=symbol,
                    from_date=price_from,
                    to_date=price_to,
                )
            except Exception as exc:
                logger.warning("retrieve_data: Finnhub %s price history failed: %s", role, exc)

        if src is None:
            continue

        src.metadata["series_role"] = role
        src.metadata["benchmark_ticker"] = Config.BENCHMARK_TICKER
        for chunk in cks:
            chunk.metadata["series_role"] = role
            chunk.metadata["benchmark_ticker"] = Config.BENCHMARK_TICKER
        new_sources.append(src)
        new_chunks.extend(cks)
        logger.info("retrieve_data: %s price history – %d chunks", role, len(cks))

    logger.info(
        "retrieve_data: complete – %d sources, %d chunks",
        len(new_sources), len(new_chunks),
    )

    updates["sources"] = new_sources
    updates["evidence_chunks"] = new_chunks
    return updates
