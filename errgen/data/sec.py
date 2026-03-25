"""
SEC EDGAR data provider.

Uses official SEC endpoints for:
  - ticker -> CIK lookup
  - company submissions metadata
  - filing document retrieval from EDGAR archives
"""

from __future__ import annotations

from calendar import monthrange
import html
import logging
import re
from datetime import date
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    no_script = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    no_tags = re.sub(r"(?s)<[^>]+>", " ", no_script)
    return _normalise_whitespace(html.unescape(no_tags))


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


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _as_of_cutoff(as_of_date: str | None) -> date | None:
    if not as_of_date:
        return None
    if len(as_of_date) == 7:
        year, month = map(int, as_of_date.split("-"))
        return date(year, month, monthrange(year, month)[1])
    return _parse_date(as_of_date)


def _period_sort_key(period: str) -> tuple[int, int, str]:
    match = re.match(r"^(Q[1-4]|FY)\s+(\d{4})$", period.strip(), flags=re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        year = int(match.group(2))
        order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}.get(label, 0)
        return (year, order, period)
    return (0, 0, period)


class SECClient(BaseDataClient):
    """Client for SEC EDGAR submissions and filing documents."""

    _ticker_cache: dict[str, str] | None = None

    def __init__(self, user_agent: str | None = None) -> None:
        self.base_url = Config.SEC_BASE_URL.rstrip("/")
        self.archives_base_url = Config.SEC_ARCHIVES_BASE_URL.rstrip("/")
        self.tickers_url = Config.SEC_TICKERS_URL
        self.user_agent = user_agent or Config.SEC_USER_AGENT

    def _sec_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }

    def _resolve_cik(self, ticker: str) -> str:
        if SECClient._ticker_cache is None:
            raw = self._get(self.tickers_url, headers=self._sec_headers())
            mapping: dict[str, str] = {}
            if isinstance(raw, dict):
                for item in raw.values():
                    if not isinstance(item, dict):
                        continue
                    symbol = str(item.get("ticker", "")).upper()
                    cik = item.get("cik_str")
                    if symbol and cik is not None:
                        mapping[symbol] = str(int(cik)).zfill(10)
            SECClient._ticker_cache = mapping

        cik = SECClient._ticker_cache.get(ticker.upper()) if SECClient._ticker_cache else None
        if not cik:
            raise ValueError(f"SEC: no CIK mapping found for ticker '{ticker}'")
        return cik

    def _companyfacts(self, ticker: str) -> tuple[str, dict[str, Any]]:
        cik = self._resolve_cik(ticker)
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        raw = self._get(url, headers=self._sec_headers())
        if not isinstance(raw, dict):
            raise ValueError(f"SEC: invalid companyfacts payload for '{ticker}'")
        return cik, raw

    @staticmethod
    def _period_label(item: dict[str, Any]) -> str | None:
        fp = str(item.get("fp") or "").upper()
        fy = item.get("fy")
        if fp and fy is not None:
            return f"{fp} {fy}"
        end = _parse_date(item.get("end"))
        return end.isoformat() if end else None

    def _companyfacts_metric(
        self,
        companyfacts: dict[str, Any],
        concepts: list[tuple[str, list[str]]],
        *,
        period: str,
        as_of_date: str | None,
    ) -> dict[str, dict[str, Any]]:
        facts_root = companyfacts.get("facts", {}).get("us-gaap", {})
        cutoff = _as_of_cutoff(as_of_date)
        selected: dict[str, dict[str, Any]] = {}

        for concept, units in concepts:
            entry = facts_root.get(concept) or {}
            units_map = entry.get("units") or {}
            for unit in units:
                for item in units_map.get(unit, []):
                    if not isinstance(item, dict):
                        continue
                    period_label = self._period_label(item)
                    if not period_label:
                        continue
                    fp = str(item.get("fp") or "").upper()
                    if period == "quarter" and fp not in {"Q1", "Q2", "Q3", "Q4"}:
                        continue
                    if period == "annual" and fp not in {"FY"}:
                        continue
                    form = str(item.get("form") or "").upper()
                    if form and form not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
                        continue
                    end_date = _parse_date(item.get("end"))
                    start_date = _parse_date(item.get("start"))
                    filed_date = _parse_date(item.get("filed"))
                    if cutoff and ((end_date and end_date > cutoff) or (filed_date and filed_date > cutoff)):
                        continue
                    if start_date and end_date:
                        duration_days = (end_date - start_date).days
                        if period == "quarter" and not (70 <= duration_days <= 110):
                            continue
                        if period == "annual" and duration_days < 300:
                            continue
                    value = item.get("val")
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue

                    current = selected.get(period_label)
                    candidate = {
                        "concept": concept,
                        "unit": unit,
                        "value": numeric_value,
                        "start": start_date,
                        "end": end_date,
                        "filed": filed_date,
                        "form": form,
                        "fy": item.get("fy"),
                        "fp": fp,
                        "frame": item.get("frame"),
                    }
                    if current is None:
                        selected[period_label] = candidate
                        continue

                    current_filed = current.get("filed") or date.min
                    candidate_filed = candidate.get("filed") or date.min
                    current_end = current.get("end") or date.min
                    candidate_end = candidate.get("end") or date.min
                    if (candidate_filed, candidate_end) >= (current_filed, current_end):
                        selected[period_label] = candidate

        return selected

    def _statement_source(
        self,
        *,
        source_type: SourceType,
        ticker: str,
        cik: str,
        document_identifier: str,
        period: str,
    ) -> SourceMetadata:
        return SourceMetadata(
            source_type=source_type,
            api_source="sec_companyfacts",
            document_identifier=document_identifier,
            url=f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json",
            ticker=ticker,
            metadata={
                "endpoint": "api/xbrl/companyfacts",
                "ticker": ticker,
                "cik": cik,
                "period": period,
            },
        )

    def get_income_statement(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
        as_of_date: str | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        cik, companyfacts = self._companyfacts(ticker)

        metrics = {
            "revenue": self._companyfacts_metric(
                companyfacts,
                [
                    ("RevenueFromContractWithCustomerExcludingAssessedTax", ["USD"]),
                    ("SalesRevenueNet", ["USD"]),
                    ("Revenues", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "grossProfit": self._companyfacts_metric(
                companyfacts,
                [("GrossProfit", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "operatingIncome": self._companyfacts_metric(
                companyfacts,
                [("OperatingIncomeLoss", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "netIncome": self._companyfacts_metric(
                companyfacts,
                [("NetIncomeLoss", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "eps": self._companyfacts_metric(
                companyfacts,
                [("EarningsPerShareDiluted", ["USD/shares", "USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "researchAndDevelopmentExpenses": self._companyfacts_metric(
                companyfacts,
                [("ResearchAndDevelopmentExpense", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
        }

        periods = sorted(
            {period_label for values in metrics.values() for period_label in values},
            key=_period_sort_key,
            reverse=True,
        )[:limit]
        if not periods:
            raise ValueError(f"SEC companyfacts returned no income statement periods for '{ticker}'")

        source = self._statement_source(
            source_type=SourceType.INCOME_STATEMENT,
            ticker=ticker,
            cik=cik,
            document_identifier=f"sec_companyfacts_income_{ticker}_{period}",
            period=period,
        )

        chunks: list[EvidenceChunk] = []
        for period_label in periods:
            revenue = metrics["revenue"].get(period_label, {}).get("value")
            gross_profit = metrics["grossProfit"].get(period_label, {}).get("value")
            op_income = metrics["operatingIncome"].get(period_label, {}).get("value")
            net_income = metrics["netIncome"].get(period_label, {}).get("value")
            eps = metrics["eps"].get(period_label, {}).get("value")
            rd = metrics["researchAndDevelopmentExpenses"].get(period_label, {}).get("value")
            gross_margin = (gross_profit / revenue) if revenue not in (None, 0) and gross_profit is not None else None
            op_margin = (op_income / revenue) if revenue not in (None, 0) and op_income is not None else None
            net_margin = (net_income / revenue) if revenue not in (None, 0) and net_income is not None else None
            date_str = metrics["revenue"].get(period_label, {}).get("end")
            date_text = date_str.isoformat() if isinstance(date_str, date) else "N/A"

            summary_lines = [
                f"Income Statement – {ticker} | Period: {period_label} | Date: {date_text}",
                f"  Revenue:            {_fmt_usd(revenue)}",
                f"  Gross Profit:       {_fmt_usd(gross_profit)} (margin {_fmt_pct(gross_margin)})",
                f"  Operating Income:   {_fmt_usd(op_income)} (margin {_fmt_pct(op_margin)})",
                f"  Net Income:         {_fmt_usd(net_income)} (margin {_fmt_pct(net_margin)})",
                f"  EBITDA:             N/A",
                f"  EPS:                {eps if eps is not None else 'N/A'}",
                f"  R&D Expenses:       {_fmt_usd(rd)}",
            ]
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.INCOME_STATEMENT,
                    text="\n".join(summary_lines),
                    period=period_label,
                    field_name="income_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_text},
                )
            )

            numeric_fields = {
                "revenue": ("Revenue", revenue, "USD"),
                "grossProfit": ("Gross Profit", gross_profit, "USD"),
                "grossProfitRatio": ("Gross Margin", gross_margin, "ratio"),
                "operatingIncome": ("Operating Income", op_income, "USD"),
                "operatingIncomeRatio": ("Operating Margin", op_margin, "ratio"),
                "netIncome": ("Net Income", net_income, "USD"),
                "netIncomeRatio": ("Net Margin", net_margin, "ratio"),
                "eps": ("EPS (Diluted)", eps, "USD"),
                "researchAndDevelopmentExpenses": ("R&D Expenses", rd, "USD"),
            }
            for field_key, (label, value, unit) in numeric_fields.items():
                if value is None:
                    continue
                display = _fmt_pct(value) if unit == "ratio" else _fmt_usd(value) if unit == "USD" else str(value)
                chunks.append(
                    EvidenceChunk(
                        source_id=source.source_id,
                        source_type=SourceType.INCOME_STATEMENT,
                        text=f"{ticker} {label} for {period_label}: {display}",
                        period=period_label,
                        field_name=field_key,
                        numeric_value=float(value),
                        unit=unit,
                        metadata={"ticker": ticker, "period": period_label, "field": field_key},
                    )
                )

        logger.info("SEC companyfacts: fetched income statement for %s (%s, %d chunks)", ticker, period, len(chunks))
        return source, chunks

    def get_balance_sheet(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
        as_of_date: str | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        cik, companyfacts = self._companyfacts(ticker)

        metrics = {
            "totalAssets": self._companyfacts_metric(
                companyfacts,
                [("Assets", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "totalLiabilities": self._companyfacts_metric(
                companyfacts,
                [("Liabilities", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "totalStockholdersEquity": self._companyfacts_metric(
                companyfacts,
                [
                    ("StockholdersEquity", ["USD"]),
                    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "cashAndCashEquivalents": self._companyfacts_metric(
                companyfacts,
                [
                    ("CashAndCashEquivalentsAtCarryingValue", ["USD"]),
                    ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "totalDebt": self._companyfacts_metric(
                companyfacts,
                [
                    ("DebtAndFinanceLeaseObligations", ["USD"]),
                    ("Debt", ["USD"]),
                    ("LongTermDebtAndFinanceLeaseObligations", ["USD"]),
                    ("LongTermDebt", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "longTermDebt": self._companyfacts_metric(
                companyfacts,
                [
                    ("LongTermDebtAndFinanceLeaseObligations", ["USD"]),
                    ("LongTermDebtAndCapitalLeaseObligations", ["USD"]),
                    ("LongTermDebt", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "totalCurrentAssets": self._companyfacts_metric(
                companyfacts,
                [("AssetsCurrent", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "totalCurrentLiabilities": self._companyfacts_metric(
                companyfacts,
                [("LiabilitiesCurrent", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
        }

        periods = sorted(
            {period_label for values in metrics.values() for period_label in values},
            key=_period_sort_key,
            reverse=True,
        )[:limit]
        if not periods:
            raise ValueError(f"SEC companyfacts returned no balance sheet periods for '{ticker}'")

        source = self._statement_source(
            source_type=SourceType.BALANCE_SHEET,
            ticker=ticker,
            cik=cik,
            document_identifier=f"sec_companyfacts_balance_{ticker}_{period}",
            period=period,
        )

        chunks: list[EvidenceChunk] = []
        for period_label in periods:
            total_assets = metrics["totalAssets"].get(period_label, {}).get("value")
            total_liabilities = metrics["totalLiabilities"].get(period_label, {}).get("value")
            total_equity = metrics["totalStockholdersEquity"].get(period_label, {}).get("value")
            cash = metrics["cashAndCashEquivalents"].get(period_label, {}).get("value")
            total_debt = metrics["totalDebt"].get(period_label, {}).get("value")
            long_term_debt = metrics["longTermDebt"].get(period_label, {}).get("value")
            current_assets = metrics["totalCurrentAssets"].get(period_label, {}).get("value")
            current_liabilities = metrics["totalCurrentLiabilities"].get(period_label, {}).get("value")
            date_str = metrics["totalAssets"].get(period_label, {}).get("end")
            date_text = date_str.isoformat() if isinstance(date_str, date) else "N/A"

            summary_lines = [
                f"Balance Sheet – {ticker} | Period: {period_label} | Date: {date_text}",
                f"  Total Assets:              {_fmt_usd(total_assets)}",
                f"  Total Liabilities:         {_fmt_usd(total_liabilities)}",
                f"  Total Stockholders' Equity:{_fmt_usd(total_equity)}",
                f"  Cash & Equivalents:        {_fmt_usd(cash)}",
                f"  Short-Term Investments:    N/A",
                f"  Total Debt:                {_fmt_usd(total_debt)}",
                f"  Long-Term Debt:            {_fmt_usd(long_term_debt)}",
                f"  Current Assets:            {_fmt_usd(current_assets)}",
                f"  Current Liabilities:       {_fmt_usd(current_liabilities)}",
            ]
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.BALANCE_SHEET,
                    text="\n".join(summary_lines),
                    period=period_label,
                    field_name="balance_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_text},
                )
            )

            numeric_fields = {
                "totalAssets": ("Total Assets", total_assets),
                "totalLiabilities": ("Total Liabilities", total_liabilities),
                "totalStockholdersEquity": ("Stockholders' Equity", total_equity),
                "cashAndCashEquivalents": ("Cash & Equivalents", cash),
                "totalDebt": ("Total Debt", total_debt),
                "longTermDebt": ("Long-Term Debt", long_term_debt),
                "totalCurrentAssets": ("Total Current Assets", current_assets),
                "totalCurrentLiabilities": ("Total Current Liabilities", current_liabilities),
            }
            for field_key, (label, value) in numeric_fields.items():
                if value is None:
                    continue
                chunks.append(
                    EvidenceChunk(
                        source_id=source.source_id,
                        source_type=SourceType.BALANCE_SHEET,
                        text=f"{ticker} {label} for {period_label}: {_fmt_usd(value)}",
                        period=period_label,
                        field_name=field_key,
                        numeric_value=float(value),
                        unit="USD",
                        metadata={"ticker": ticker, "period": period_label, "field": field_key},
                    )
                )

        logger.info("SEC companyfacts: fetched balance sheet for %s (%s, %d chunks)", ticker, period, len(chunks))
        return source, chunks

    def get_cash_flow(
        self,
        ticker: str,
        period: str = "annual",
        limit: int | None = None,
        as_of_date: str | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        limit = limit or Config.MAX_FINANCIAL_PERIODS
        cik, companyfacts = self._companyfacts(ticker)

        metrics = {
            "operatingCashFlow": self._companyfacts_metric(
                companyfacts,
                [
                    ("NetCashProvidedByUsedInOperatingActivities", ["USD"]),
                    ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "capitalExpenditure": self._companyfacts_metric(
                companyfacts,
                [
                    ("PaymentsToAcquirePropertyPlantAndEquipment", ["USD"]),
                    ("CapitalExpendituresIncurredButNotYetPaid", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
            "dividendsPaid": self._companyfacts_metric(
                companyfacts,
                [("PaymentsOfDividends", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "commonStockRepurchased": self._companyfacts_metric(
                companyfacts,
                [("PaymentsForRepurchaseOfCommonStock", ["USD"])],
                period=period,
                as_of_date=as_of_date,
            ),
            "depreciationAndAmortization": self._companyfacts_metric(
                companyfacts,
                [
                    ("DepreciationDepletionAndAmortization", ["USD"]),
                    ("DepreciationAmortizationAndAccretionNet", ["USD"]),
                ],
                period=period,
                as_of_date=as_of_date,
            ),
        }

        periods = sorted(
            {period_label for values in metrics.values() for period_label in values},
            key=_period_sort_key,
            reverse=True,
        )[:limit]
        if not periods:
            raise ValueError(f"SEC companyfacts returned no cash flow periods for '{ticker}'")

        source = self._statement_source(
            source_type=SourceType.CASH_FLOW,
            ticker=ticker,
            cik=cik,
            document_identifier=f"sec_companyfacts_cashflow_{ticker}_{period}",
            period=period,
        )

        chunks: list[EvidenceChunk] = []
        for period_label in periods:
            op_cf = metrics["operatingCashFlow"].get(period_label, {}).get("value")
            capex = metrics["capitalExpenditure"].get(period_label, {}).get("value")
            div = metrics["dividendsPaid"].get(period_label, {}).get("value")
            buyback = metrics["commonStockRepurchased"].get(period_label, {}).get("value")
            dep = metrics["depreciationAndAmortization"].get(period_label, {}).get("value")
            free_cf = None
            if op_cf is not None and capex is not None:
                free_cf = op_cf + capex if capex < 0 else op_cf - capex
            date_str = metrics["operatingCashFlow"].get(period_label, {}).get("end")
            date_text = date_str.isoformat() if isinstance(date_str, date) else "N/A"

            summary_lines = [
                f"Cash Flow Statement – {ticker} | Period: {period_label} | Date: {date_text}",
                f"  Operating Cash Flow:     {_fmt_usd(op_cf)}",
                f"  Capital Expenditure:     {_fmt_usd(capex)}",
                f"  Free Cash Flow:          {_fmt_usd(free_cf)}",
                f"  Dividends Paid:          {_fmt_usd(div)}",
                f"  Stock Buybacks:          {_fmt_usd(buyback)}",
                f"  Depreciation & Amort.:   {_fmt_usd(dep)}",
            ]
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.CASH_FLOW,
                    text="\n".join(summary_lines),
                    period=period_label,
                    field_name="cashflow_summary",
                    metadata={"ticker": ticker, "period": period_label, "date": date_text},
                )
            )

            numeric_fields = {
                "operatingCashFlow": ("Operating Cash Flow", op_cf),
                "capitalExpenditure": ("Capital Expenditure", capex),
                "freeCashFlow": ("Free Cash Flow", free_cf),
                "dividendsPaid": ("Dividends Paid", div),
                "commonStockRepurchased": ("Stock Buybacks", buyback),
                "depreciationAndAmortization": ("D&A", dep),
            }
            for field_key, (label, value) in numeric_fields.items():
                if value is None:
                    continue
                chunks.append(
                    EvidenceChunk(
                        source_id=source.source_id,
                        source_type=SourceType.CASH_FLOW,
                        text=f"{ticker} {label} for {period_label}: {_fmt_usd(value)}",
                        period=period_label,
                        field_name=field_key,
                        numeric_value=float(value),
                        unit="USD",
                        metadata={"ticker": ticker, "period": period_label, "field": field_key},
                    )
                )

        logger.info("SEC companyfacts: fetched cash flow for %s (%s, %d chunks)", ticker, period, len(chunks))
        return source, chunks

    @staticmethod
    def _extract_filing_sections(text: str) -> list[tuple[str, str]]:
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

    @staticmethod
    def _recent_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
        recent = submissions.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []

        def _value(field: str, idx: int, total: int) -> Any:
            values = recent.get(field) or [None] * total
            if idx >= len(values):
                return None
            return values[idx]

        forms = recent.get("form") or []
        total = len(forms)
        records: list[dict[str, Any]] = []
        for idx in range(total):
            records.append(
                {
                    "form": forms[idx],
                    "filingDate": _value("filingDate", idx, total),
                    "acceptanceDateTime": _value("acceptanceDateTime", idx, total),
                    "accessionNumber": _value("accessionNumber", idx, total),
                    "primaryDocument": _value("primaryDocument", idx, total),
                    "primaryDocDescription": _value("primaryDocDescription", idx, total),
                    "reportDate": _value("reportDate", idx, total),
                }
            )
        return records

    def get_sec_filings(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch recent 10-K / 10-Q filings from SEC EDGAR and extract summary chunks.
        """
        limit = limit or Config.MAX_SEC_FILINGS
        cik = self._resolve_cik(ticker)
        submissions_url = f"{self.base_url}/submissions/CIK{cik}.json"
        submissions = self._get(submissions_url, headers=self._sec_headers())

        filings = []
        for item in self._recent_filings(submissions):
            form_type = (item.get("form") or "").upper()
            filed_at = item.get("filingDate") or ""
            if form_type not in {"10-K", "10-Q"}:
                continue
            if filed_at and (filed_at < from_date or filed_at > to_date):
                continue
            filings.append(item)
            if len(filings) >= limit:
                break

        source = SourceMetadata(
            source_type=SourceType.FILING,
            api_source="sec",
            document_identifier=f"sec_filings_{ticker}_{from_date}_{to_date}",
            url=submissions_url,
            ticker=ticker,
            metadata={
                "endpoint": "submissions/CIK.json",
                "ticker": ticker,
                "cik": cik,
                "from": from_date,
                "to": to_date,
                "n_filings": len(filings),
            },
        )

        chunks: list[EvidenceChunk] = []
        cik_numeric = str(int(cik))
        for filing in filings:
            form_type = (filing.get("form") or "").upper()
            filed_at = filing.get("filingDate") or ""
            accepted_at = filing.get("acceptanceDateTime") or ""
            accession = filing.get("accessionNumber") or ""
            primary_document = filing.get("primaryDocument") or ""
            filing_title = filing.get("primaryDocDescription") or f"{ticker} {form_type} filing"
            accession_compact = accession.replace("-", "")
            final_link = ""
            if accession_compact and primary_document:
                final_link = (
                    f"{self.archives_base_url}/{cik_numeric}/"
                    f"{accession_compact}/{primary_document}"
                )

            metadata = {
                "ticker": ticker,
                "cik": cik,
                "form_type": form_type,
                "filed_at": filed_at,
                "accepted_at": accepted_at,
                "accession_number": accession,
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
                raw_text = self._get_text(final_link, headers=self._sec_headers())
                cleaned = _strip_html(raw_text)
            except Exception as exc:
                logger.warning("SEC: failed to fetch filing body for %s %s: %s", ticker, form_type, exc)
                continue

            if not cleaned:
                continue

            for section_name, excerpt in self._extract_filing_sections(cleaned):
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

        logger.info("SEC: fetched %d filing chunks for %s", len(chunks), ticker)
        return source, chunks
