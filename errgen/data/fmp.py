"""
Financial Modeling Prep (FMP) data provider.

Endpoints used:
  /v3/profile/{ticker}              – company profile & metadata
  /v3/income-statement/{ticker}     – income statement (annual / quarterly)
  /v3/balance-sheet-statement/{ticker}
  /v3/cash-flow-statement/{ticker}
  /v3/stock_news                    – company-specific news from FMP
  /v3/historical-price-full/{ticker} – daily OHLCV (optional)

Registration: https://financialmodelingprep.com/developer/docs/
Free tier: 250 API calls/day.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


def _fmt_usd(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"${float(val):,.0f}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2%}"
    except (TypeError, ValueError):
        return str(val)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    no_script = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    no_tags = re.sub(r"(?s)<[^>]+>", " ", no_script)
    return _normalise_whitespace(html.unescape(no_tags))


def _coerce_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


class FMPClient(BaseDataClient):
    """
    Client for Financial Modeling Prep API.

    Each public method returns a (SourceMetadata, list[EvidenceChunk]) pair
    so the caller can track provenance precisely.

    IMPORTANT: This client raises ValueError immediately if FMP_API_KEY is
    missing.  Do NOT silently fall back to fake data.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or Config.FMP_API_KEY
        if not self.api_key:
            raise ValueError(
                "FMP_API_KEY is not set.\n"
                "Register at https://financialmodelingprep.com/developer/docs/ "
                "(free tier: 250 requests/day) and set FMP_API_KEY in .env"
            )
        self.base_url = Config.FMP_BASE_URL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fmp_get(self, endpoint: str, extra_params: dict | None = None) -> Any:
        params: dict[str, Any] = {"apikey": self.api_key}
        if extra_params:
            params.update(extra_params)
        url = f"{self.base_url}/{endpoint}"
        data = self._get(url, params=params)
        # FMP sometimes wraps errors in a dict
        if isinstance(data, dict) and "Error Message" in data:
            raise ValueError(f"FMP API error for {endpoint}: {data['Error Message']}")
        return data

    # ------------------------------------------------------------------
    # Company profile
    # ------------------------------------------------------------------

    def get_company_profile(
        self, ticker: str
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """Fetch company overview, sector, industry, and business description."""
        raw = self._fmp_get("profile", {"symbol": ticker})
        if not raw:
            raise ValueError(
                f"FMP returned no company profile for ticker '{ticker}'. "
                "Verify the ticker is valid and your FMP_API_KEY has access."
            )

        profile: dict = raw[0] if isinstance(raw, list) else raw

        source = SourceMetadata(
            source_type=SourceType.COMPANY_PROFILE,
            api_source="fmp",
            document_identifier=f"fmp_profile_{ticker}",
            url=f"https://financialmodelingprep.com/financial-summary/{ticker}",
            ticker=ticker,
            metadata={"endpoint": "profile", "symbol": ticker},
        )

        chunks: list[EvidenceChunk] = []

        # Overview chunk
        mkt_cap = profile.get("mktCap")
        overview_text = (
            f"{profile.get('companyName', ticker)} (Ticker: {ticker}) | "
            f"Exchange: {profile.get('exchange', 'N/A')} | "
            f"Sector: {profile.get('sector', 'N/A')} | "
            f"Industry: {profile.get('industry', 'N/A')} | "
            f"Market Cap: {_fmt_usd(mkt_cap)} | "
            f"CEO: {profile.get('ceo', 'N/A')} | "
            f"Full-Time Employees: {profile.get('fullTimeEmployees', 'N/A')} | "
            f"IPO Date: {profile.get('ipoDate', 'N/A')} | "
            f"Country: {profile.get('country', 'N/A')} | "
            f"Website: {profile.get('website', 'N/A')}"
        )
        chunks.append(
            EvidenceChunk(
                source_id=source.source_id,
                source_type=SourceType.COMPANY_PROFILE,
                text=overview_text,
                field_name="overview",
                metadata={"ticker": ticker, "field": "overview"},
            )
        )

        # Business description chunk (can be long – keep up to 3 000 chars)
        desc = profile.get("description", "").strip()
        if desc:
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.COMPANY_PROFILE,
                    text=desc[:3000],
                    field_name="description",
                    metadata={"ticker": ticker, "field": "description"},
                )
            )

        logger.info("FMP: fetched company profile for %s (%d chunks)", ticker, len(chunks))
        return source, chunks

    # ------------------------------------------------------------------
    # Income statement
    # ------------------------------------------------------------------

    def get_income_statement(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch income statement data.

        period: "annual" | "quarter"
        limit:  number of periods to retrieve (default: Config.MAX_FINANCIAL_PERIODS)
        """
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        raw = self._fmp_get(
            "income-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )
        if not raw:
            raise ValueError(
                f"FMP returned no income statement for '{ticker}' (period={period})."
            )

        source = SourceMetadata(
            source_type=SourceType.INCOME_STATEMENT,
            api_source="fmp",
            document_identifier=f"fmp_income_{ticker}_{period}",
            ticker=ticker,
            metadata={
                "endpoint": "income-statement",
                "symbol": ticker,
                "period": period,
            },
        )

        chunks = self._income_to_chunks(ticker, raw, source.source_id)
        logger.info(
            "FMP: fetched income statement for %s (%s, %d periods → %d chunks)",
            ticker, period, len(raw), len(chunks),
        )
        return source, chunks

    @staticmethod
    def _income_to_chunks(
        ticker: str, statements: list[dict], source_id: str
    ) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for stmt in statements:
            period_label = f"{stmt.get('period', '')} {stmt.get('calendarYear', '')}".strip()
            date_str = stmt.get("date", "")

            # Summary chunk with all key metrics as prose + table
            revenue = stmt.get("revenue")
            gross_profit = stmt.get("grossProfit")
            op_income = stmt.get("operatingIncome")
            net_income = stmt.get("netIncome")
            ebitda = stmt.get("ebitda")
            eps = stmt.get("eps")
            rd = stmt.get("researchAndDevelopmentExpenses")
            gross_margin = stmt.get("grossProfitRatio")
            op_margin = stmt.get("operatingIncomeRatio")
            net_margin = stmt.get("netIncomeRatio")

            lines = [
                f"Income Statement – {ticker} | Period: {period_label} | Date: {date_str}",
                f"  Revenue:            {_fmt_usd(revenue)}",
                f"  Gross Profit:       {_fmt_usd(gross_profit)} (margin {_fmt_pct(gross_margin)})",
                f"  Operating Income:   {_fmt_usd(op_income)} (margin {_fmt_pct(op_margin)})",
                f"  Net Income:         {_fmt_usd(net_income)} (margin {_fmt_pct(net_margin)})",
                f"  EBITDA:             {_fmt_usd(ebitda)}",
                f"  EPS:                {eps if eps is not None else 'N/A'}",
                f"  R&D Expenses:       {_fmt_usd(rd)}",
            ]
            summary_text = "\n".join(lines)

            chunks.append(
                EvidenceChunk(
                    source_id=source_id,
                    source_type=SourceType.INCOME_STATEMENT,
                    text=summary_text,
                    period=period_label,
                    field_name="income_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_str},
                )
            )

            # Individual numeric chunks for precise citation
            numeric_fields = {
                "revenue": ("Revenue", revenue, "USD"),
                "grossProfit": ("Gross Profit", gross_profit, "USD"),
                "grossProfitRatio": ("Gross Margin", gross_margin, "ratio"),
                "operatingIncome": ("Operating Income", op_income, "USD"),
                "operatingIncomeRatio": ("Operating Margin", op_margin, "ratio"),
                "netIncome": ("Net Income", net_income, "USD"),
                "netIncomeRatio": ("Net Margin", net_margin, "ratio"),
                "ebitda": ("EBITDA", ebitda, "USD"),
                "eps": ("EPS (Diluted)", eps, "USD"),
                "researchAndDevelopmentExpenses": ("R&D Expenses", rd, "USD"),
            }
            for field_key, (label, value, unit) in numeric_fields.items():
                if value is not None:
                    if unit == "USD":
                        display = _fmt_usd(value)
                    elif unit == "ratio":
                        display = _fmt_pct(value)
                    else:
                        display = str(value)
                    chunks.append(
                        EvidenceChunk(
                            source_id=source_id,
                            source_type=SourceType.INCOME_STATEMENT,
                            text=(
                                f"{ticker} {label} for {period_label}: {display}"
                            ),
                            period=period_label,
                            field_name=field_key,
                            numeric_value=float(value),
                            unit=unit,
                            metadata={
                                "ticker": ticker,
                                "period": period_label,
                                "field": field_key,
                            },
                        )
                    )

        return chunks

    # ------------------------------------------------------------------
    # Balance sheet
    # ------------------------------------------------------------------

    def get_balance_sheet(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        raw = self._fmp_get(
            "balance-sheet-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )
        if not raw:
            raise ValueError(
                f"FMP returned no balance sheet for '{ticker}' (period={period})."
            )

        source = SourceMetadata(
            source_type=SourceType.BALANCE_SHEET,
            api_source="fmp",
            document_identifier=f"fmp_balance_{ticker}_{period}",
            ticker=ticker,
            metadata={
                "endpoint": "balance-sheet-statement",
                "symbol": ticker,
                "period": period,
            },
        )

        chunks = self._balance_to_chunks(ticker, raw, source.source_id)
        logger.info(
            "FMP: fetched balance sheet for %s (%s, %d periods → %d chunks)",
            ticker, period, len(raw), len(chunks),
        )
        return source, chunks

    @staticmethod
    def _balance_to_chunks(
        ticker: str, statements: list[dict], source_id: str
    ) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for stmt in statements:
            period_label = f"{stmt.get('period', '')} {stmt.get('calendarYear', '')}".strip()
            date_str = stmt.get("date", "")

            total_assets = stmt.get("totalAssets")
            total_liabilities = stmt.get("totalLiabilities")
            total_equity = stmt.get("totalStockholdersEquity")
            cash = stmt.get("cashAndCashEquivalents")
            short_term_inv = stmt.get("shortTermInvestments")
            total_debt = stmt.get("totalDebt")
            long_term_debt = stmt.get("longTermDebt")
            current_ratio_num = stmt.get("totalCurrentAssets")
            current_ratio_den = stmt.get("totalCurrentLiabilities")

            lines = [
                f"Balance Sheet – {ticker} | Period: {period_label} | Date: {date_str}",
                f"  Total Assets:              {_fmt_usd(total_assets)}",
                f"  Total Liabilities:         {_fmt_usd(total_liabilities)}",
                f"  Total Stockholders' Equity:{_fmt_usd(total_equity)}",
                f"  Cash & Equivalents:        {_fmt_usd(cash)}",
                f"  Short-Term Investments:    {_fmt_usd(short_term_inv)}",
                f"  Total Debt:                {_fmt_usd(total_debt)}",
                f"  Long-Term Debt:            {_fmt_usd(long_term_debt)}",
                f"  Current Assets:            {_fmt_usd(current_ratio_num)}",
                f"  Current Liabilities:       {_fmt_usd(current_ratio_den)}",
            ]
            chunks.append(
                EvidenceChunk(
                    source_id=source_id,
                    source_type=SourceType.BALANCE_SHEET,
                    text="\n".join(lines),
                    period=period_label,
                    field_name="balance_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_str},
                )
            )

            numeric_fields = {
                "totalAssets": ("Total Assets", total_assets, "USD"),
                "totalLiabilities": ("Total Liabilities", total_liabilities, "USD"),
                "totalStockholdersEquity": ("Stockholders' Equity", total_equity, "USD"),
                "cashAndCashEquivalents": ("Cash & Equivalents", cash, "USD"),
                "totalDebt": ("Total Debt", total_debt, "USD"),
                "longTermDebt": ("Long-Term Debt", long_term_debt, "USD"),
                "totalCurrentAssets": ("Total Current Assets", current_ratio_num, "USD"),
                "totalCurrentLiabilities": (
                    "Total Current Liabilities", current_ratio_den, "USD"
                ),
            }
            for field_key, (label, value, unit) in numeric_fields.items():
                if value is not None:
                    chunks.append(
                        EvidenceChunk(
                            source_id=source_id,
                            source_type=SourceType.BALANCE_SHEET,
                            text=f"{ticker} {label} for {period_label}: {_fmt_usd(value)}",
                            period=period_label,
                            field_name=field_key,
                            numeric_value=float(value),
                            unit=unit,
                            metadata={
                                "ticker": ticker,
                                "period": period_label,
                                "field": field_key,
                            },
                        )
                    )

        return chunks

    # ------------------------------------------------------------------
    # Cash flow statement
    # ------------------------------------------------------------------

    def get_cash_flow(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        raw = self._fmp_get(
            "cash-flow-statement",
            {"symbol": ticker, "period": period, "limit": limit},
        )
        if not raw:
            raise ValueError(
                f"FMP returned no cash flow statement for '{ticker}' (period={period})."
            )

        source = SourceMetadata(
            source_type=SourceType.CASH_FLOW,
            api_source="fmp",
            document_identifier=f"fmp_cashflow_{ticker}_{period}",
            ticker=ticker,
            metadata={
                "endpoint": "cash-flow-statement",
                "symbol": ticker,
                "period": period,
            },
        )

        chunks = self._cashflow_to_chunks(ticker, raw, source.source_id)
        logger.info(
            "FMP: fetched cash flow for %s (%s, %d periods → %d chunks)",
            ticker, period, len(raw), len(chunks),
        )
        return source, chunks

    @staticmethod
    def _cashflow_to_chunks(
        ticker: str, statements: list[dict], source_id: str
    ) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for stmt in statements:
            period_label = f"{stmt.get('period', '')} {stmt.get('calendarYear', '')}".strip()
            date_str = stmt.get("date", "")

            op_cf = stmt.get("operatingCashFlow") or stmt.get("netCashProvidedByOperatingActivities")
            capex = stmt.get("capitalExpenditure")
            free_cf = stmt.get("freeCashFlow")
            div = stmt.get("dividendsPaid")
            buyback = stmt.get("commonStockRepurchased")
            dep = stmt.get("depreciationAndAmortization")

            lines = [
                f"Cash Flow Statement – {ticker} | Period: {period_label} | Date: {date_str}",
                f"  Operating Cash Flow:     {_fmt_usd(op_cf)}",
                f"  Capital Expenditure:     {_fmt_usd(capex)}",
                f"  Free Cash Flow:          {_fmt_usd(free_cf)}",
                f"  Dividends Paid:          {_fmt_usd(div)}",
                f"  Stock Buybacks:          {_fmt_usd(buyback)}",
                f"  Depreciation & Amort.:   {_fmt_usd(dep)}",
            ]
            chunks.append(
                EvidenceChunk(
                    source_id=source_id,
                    source_type=SourceType.CASH_FLOW,
                    text="\n".join(lines),
                    period=period_label,
                    field_name="cashflow_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_str},
                )
            )

            numeric_fields = {
                "operatingCashFlow": ("Operating Cash Flow", op_cf, "USD"),
                "capitalExpenditure": ("Capital Expenditure", capex, "USD"),
                "freeCashFlow": ("Free Cash Flow", free_cf, "USD"),
                "dividendsPaid": ("Dividends Paid", div, "USD"),
                "commonStockRepurchased": ("Stock Buybacks", buyback, "USD"),
                "depreciationAndAmortization": ("D&A", dep, "USD"),
            }
            for field_key, (label, value, unit) in numeric_fields.items():
                if value is not None:
                    chunks.append(
                        EvidenceChunk(
                            source_id=source_id,
                            source_type=SourceType.CASH_FLOW,
                            text=f"{ticker} {label} for {period_label}: {_fmt_usd(value)}",
                            period=period_label,
                            field_name=field_key,
                            numeric_value=float(value),
                            unit=unit,
                            metadata={
                                "ticker": ticker,
                                "period": period_label,
                                "field": field_key,
                            },
                        )
                    )

        return chunks

    # ------------------------------------------------------------------
    # SEC filings
    # ------------------------------------------------------------------

    def get_sec_filings(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch recent SEC filings and build filing-summary / section chunks.

        The FMP filings API returns links to the primary filing documents.
        We fetch the filing HTML and extract lightweight text snippets for
        MD&A / Business / Risk Factors when possible.
        """
        limit = limit or Config.MAX_SEC_FILINGS
        raw = self._fmp_get(
            "sec-filings-search/symbol",
            {
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "page": 0,
                "limit": limit,
            },
        )

        filings = raw if isinstance(raw, list) else []
        filings = [
            item for item in filings
            if (item.get("formType") or item.get("type") or "").upper() in {"10-K", "10-Q"}
        ][:limit]

        source = SourceMetadata(
            source_type=SourceType.FILING,
            api_source="fmp",
            document_identifier=f"fmp_sec_filings_{ticker}_{from_date}_{to_date}",
            ticker=ticker,
            metadata={
                "endpoint": "sec-filings-search/symbol",
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "n_filings": len(filings),
            },
        )

        chunks: list[EvidenceChunk] = []
        for filing in filings:
            form_type = (filing.get("formType") or filing.get("type") or "").upper()
            filed_at = filing.get("filingDate") or filing.get("fillingDate") or ""
            accepted_at = filing.get("acceptedDate") or ""
            final_link = filing.get("finalLink") or filing.get("link") or filing.get("url") or ""
            filing_title = filing.get("title") or f"{ticker} {form_type} filing"

            metadata = {
                "ticker": ticker,
                "form_type": form_type,
                "filed_at": filed_at,
                "accepted_at": accepted_at,
                "url": final_link,
                "filing_title": filing_title,
            }
            summary_text = (
                f"[FILING] {ticker} {form_type} | Filed: {filed_at} | Accepted: {accepted_at}\n"
                f"Title: {filing_title}\n"
                f"URL: {final_link or 'N/A'}"
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.FILING,
                    text=summary_text,
                    field_name="filing_summary",
                    period=filed_at[:10] if filed_at else None,
                    metadata=metadata,
                )
            )

            if not final_link:
                continue

            try:
                raw_text = self._get_text(final_link)
                cleaned = _strip_html(raw_text)
            except Exception as exc:
                logger.warning("FMP: failed to fetch filing body for %s %s: %s", ticker, form_type, exc)
                continue

            if not cleaned:
                continue

            for section_name, excerpt in self._extract_filing_sections(cleaned, form_type):
                chunks.append(
                    EvidenceChunk(
                        source_id=source.source_id,
                        source_type=SourceType.FILING,
                        text=excerpt,
                        field_name=section_name,
                        period=filed_at[:10] if filed_at else None,
                        metadata=metadata,
                    )
                )

        logger.info("FMP: fetched %d filing chunks for %s", len(chunks), ticker)
        return source, chunks

    @staticmethod
    def _extract_filing_sections(text: str, form_type: str) -> list[tuple[str, str]]:
        section_specs = [
            ("filing_business", [r"item\s*1\.?\s*business", r"business overview"]),
            ("filing_mda", [r"management['’]s discussion and analysis", r"item\s*2\.?\s*management"]),
            ("filing_risk_factors", [r"item\s*1a\.?\s*risk factors", r"risk factors"]),
        ]
        lower = text.lower()
        snippets: list[tuple[str, str]] = []
        for field_name, patterns in section_specs:
            start = -1
            for pattern in patterns:
                match = re.search(pattern, lower, flags=re.IGNORECASE)
                if match:
                    start = match.start()
                    break
            if start < 0:
                continue
            excerpt = text[start:start + 2400].strip()
            if excerpt:
                snippets.append((field_name, excerpt))

        if snippets:
            return snippets

        fallback = text[:2400].strip()
        return [("filing_excerpt", fallback)] if fallback else []

    # ------------------------------------------------------------------
    # News from FMP
    # ------------------------------------------------------------------

    def get_stock_news(
        self,
        ticker: str,
        limit: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """Fetch company-specific news articles from FMP."""
        limit = limit or Config.MAX_NEWS_ARTICLES
        params: dict[str, Any] = {"symbols": ticker, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        raw = self._fmp_get("news/stock", params)
        if not raw:
            logger.warning("FMP: no news found for %s", ticker)
            raw = []

        source = SourceMetadata(
            source_type=SourceType.NEWS,
            api_source="fmp",
            document_identifier=f"fmp_news_{ticker}",
            ticker=ticker,
            metadata={"endpoint": "news/stock", "ticker": ticker},
        )

        chunks: list[EvidenceChunk] = []
        for article in raw:
            pub_date = article.get("publishedDate", "")
            title = article.get("title", "")
            text = article.get("text", "")
            url = article.get("url", "")
            site = article.get("site", "")

            if not title and not text:
                continue

            chunk_text = (
                f"[NEWS] {pub_date} | {site}\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {text[:1500]}"
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.NEWS,
                    text=chunk_text,
                    metadata={
                        "ticker": ticker,
                        "published_date": pub_date,
                        "title": title,
                        "url": url,
                        "source_site": site,
                    },
                )
            )

        logger.info("FMP: fetched %d news articles for %s", len(chunks), ticker)
        return source, chunks

    # ------------------------------------------------------------------
    # Optional: historical price data
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """Fetch daily OHLCV price data (optional, for price-based analysis)."""
        raw = self._fmp_get(
            "historical-price-eod/full",
            {"symbol": ticker, "from": from_date, "to": to_date},
        )

        if isinstance(raw, list):
            historical = raw
        elif isinstance(raw, dict):
            historical = (
                raw.get("historical")
                or raw.get("prices")
                or raw.get("results")
                or []
            )
        else:
            historical = []

        historical = [
            item for item in historical
            if isinstance(item, dict) and item.get("date") and item.get("close") is not None
        ]
        historical.sort(key=lambda item: str(item.get("date")))
        if not historical:
            logger.warning("FMP: no price history for %s (%s – %s)", ticker, from_date, to_date)

        source = SourceMetadata(
            source_type=SourceType.PRICE_DATA,
            api_source="fmp",
            document_identifier=f"fmp_price_{ticker}_{from_date}_{to_date}",
            ticker=ticker,
            metadata={
                "endpoint": "historical-price-eod/full",
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "historical": historical,
            },
        )

        chunks: list[EvidenceChunk] = []
        if historical:
            first = historical[0]
            last = historical[-1]
            first_close = _coerce_float(first.get("close"))
            last_close = _coerce_float(last.get("close"))
            highs = [_coerce_float(item.get("high")) for item in historical]
            lows = [_coerce_float(item.get("low")) for item in historical]
            highs = [val for val in highs if val is not None]
            lows = [val for val in lows if val is not None]
            if first_close is None or last_close is None or not highs or not lows:
                logger.warning("FMP: incomplete close prices for %s", ticker)
                return source, chunks
            summary = (
                f"Price history for {ticker} from {from_date} to {to_date}: "
                f"Opening price on {first['date']}: ${first_close:.2f}, "
                f"Closing price on {last['date']}: ${last_close:.2f}. "
                f"52-week high: ${max(highs):.2f}, "
                f"52-week low: ${min(lows):.2f}."
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.PRICE_DATA,
                    text=summary,
                    metadata={
                        "ticker": ticker,
                        "from_date": from_date,
                        "to_date": to_date,
                        "data_points": len(historical),
                        "price_start": first_close,
                        "price_end": last_close,
                        "price_high": max(highs),
                        "price_low": min(lows),
                    },
                )
            )

        logger.info(
            "FMP: fetched price history for %s (%d data points)", ticker, len(historical)
        )
        return source, chunks
