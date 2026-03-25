"""
Financial data extractor.

Converts raw income statement, balance sheet, and cash flow evidence chunks
into structured ExtractedFact objects AND builds the set of CalculationRequest
objects for the standard financial metrics the pipeline needs.

This layer is mostly rule-based (no LLM needed for numeric extraction from
structured FMP data).  It does use the calculator to compute derived metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from errgen.calculator.finance_calc import (
    FinanceCalculator,
    build_growth_request,
    build_margin_request,
    build_yoy_table_request,
)
from errgen.models import (
    CalculationRequest,
    CalculationResult,
    EvidenceChunk,
    ExtractedFact,
    SourceType,
)

logger = logging.getLogger(__name__)


def _period_sort_key(period: str) -> tuple[int, int, str]:
    match = re.match(r"^(Q[1-4]|FY)\s+(\d{4})$", period.strip(), flags=re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        year = int(match.group(2))
        order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}.get(label, 0)
        return (year, order, period)
    return (0, 0, period)


class FinancialExtractor:
    """
    Extracts structured facts from financial evidence chunks and generates
    standard calculation requests.

    The `extract` method is designed to be called once per pipeline run
    after data collection.  It returns:
      - A list of ExtractedFact objects (each referencing the chunk_ids that
        support them).
      - A list of CalculationResult objects from running standard financial
        metrics through the deterministic calculator.
    """

    def __init__(self) -> None:
        self.calculator = FinanceCalculator()

    def extract(
        self,
        ticker: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[list[ExtractedFact], list[CalculationRequest], list[CalculationResult]]:
        """
        Main extraction entrypoint.

        Returns:
            facts: structured ExtractedFact list
            calc_requests: CalculationRequest list (for audit trail)
            calc_results: CalculationResult list (computed values)
        """
        facts: list[ExtractedFact] = []
        calc_requests: list[CalculationRequest] = []
        calc_results: list[CalculationResult] = []

        # ----------------------------------------------------------------
        # Step 1: index numeric chunks by (source_type, period, field_name)
        # ----------------------------------------------------------------
        income_by_period: dict[str, dict[str, EvidenceChunk]] = {}
        balance_by_period: dict[str, dict[str, EvidenceChunk]] = {}
        cashflow_by_period: dict[str, dict[str, EvidenceChunk]] = {}

        for chunk in chunks:
            if not chunk.period or not chunk.field_name:
                continue
            if chunk.numeric_value is None:
                continue

            period = chunk.period
            field = chunk.field_name

            if chunk.source_type == SourceType.INCOME_STATEMENT:
                income_by_period.setdefault(period, {})[field] = chunk
            elif chunk.source_type == SourceType.BALANCE_SHEET:
                balance_by_period.setdefault(period, {})[field] = chunk
            elif chunk.source_type == SourceType.CASH_FLOW:
                cashflow_by_period.setdefault(period, {})[field] = chunk

        # ----------------------------------------------------------------
        # Step 2: create ExtractedFact for each key metric per period
        # ----------------------------------------------------------------
        key_income_fields = [
            "revenue", "grossProfit", "grossProfitRatio",
            "operatingIncome", "operatingIncomeRatio",
            "netIncome", "netIncomeRatio",
            "ebitda", "eps",
            "researchAndDevelopmentExpenses",
        ]
        key_balance_fields = [
            "totalAssets", "totalLiabilities",
            "totalStockholdersEquity", "cashAndCashEquivalents",
            "totalDebt", "longTermDebt",
            "totalCurrentAssets", "totalCurrentLiabilities",
        ]
        key_cashflow_fields = [
            "operatingCashFlow", "capitalExpenditure", "freeCashFlow",
            "dividendsPaid", "commonStockRepurchased",
        ]

        for period, fields in income_by_period.items():
            for fld in key_income_fields:
                chunk = fields.get(fld)
                if chunk and chunk.numeric_value is not None:
                    facts.append(
                        ExtractedFact(
                            chunk_ids=[chunk.chunk_id],
                            fact_type=fld,
                            subject=ticker,
                            period=period,
                            value=chunk.numeric_value,
                            unit=chunk.unit or "",
                            description=chunk.text,
                        )
                    )

        for period, fields in balance_by_period.items():
            for fld in key_balance_fields:
                chunk = fields.get(fld)
                if chunk and chunk.numeric_value is not None:
                    facts.append(
                        ExtractedFact(
                            chunk_ids=[chunk.chunk_id],
                            fact_type=fld,
                            subject=ticker,
                            period=period,
                            value=chunk.numeric_value,
                            unit=chunk.unit or "",
                            description=chunk.text,
                        )
                    )

        for period, fields in cashflow_by_period.items():
            for fld in key_cashflow_fields:
                chunk = fields.get(fld)
                if chunk and chunk.numeric_value is not None:
                    facts.append(
                        ExtractedFact(
                            chunk_ids=[chunk.chunk_id],
                            fact_type=fld,
                            subject=ticker,
                            period=period,
                            value=chunk.numeric_value,
                            unit=chunk.unit or "",
                            description=chunk.text,
                        )
                    )

        # ----------------------------------------------------------------
        # Step 3: build standard calculation requests
        # ----------------------------------------------------------------
        sorted_income_periods = sorted(income_by_period.keys(), key=_period_sort_key)
        sorted_balance_periods = sorted(balance_by_period.keys(), key=_period_sort_key)
        sorted_cashflow_periods = sorted(cashflow_by_period.keys(), key=_period_sort_key)

        # YoY revenue growth table
        revenue_series = []
        for period in sorted_income_periods:
            chunk = income_by_period[period].get("revenue")
            if chunk and chunk.numeric_value is not None:
                revenue_series.append({"period": period, "value": chunk.numeric_value})
        if len(revenue_series) >= 2:
            req = build_yoy_table_request(
                description=f"{ticker} Revenue YoY Growth Table",
                series=revenue_series,
                chunk_ids=[
                    income_by_period[p]["revenue"].chunk_id
                    for p in sorted_income_periods
                    if "revenue" in income_by_period[p]
                ],
            )
            calc_requests.append(req)
            calc_results.append(self.calculator.compute(req))

        # YoY net income growth table
        ni_series = []
        for period in sorted_income_periods:
            chunk = income_by_period[period].get("netIncome")
            if chunk and chunk.numeric_value is not None:
                ni_series.append({"period": period, "value": chunk.numeric_value})
        if len(ni_series) >= 2:
            req = build_yoy_table_request(
                description=f"{ticker} Net Income YoY Growth Table",
                series=ni_series,
                chunk_ids=[
                    income_by_period[p]["netIncome"].chunk_id
                    for p in sorted_income_periods
                    if "netIncome" in income_by_period[p]
                ],
            )
            calc_requests.append(req)
            calc_results.append(self.calculator.compute(req))

        # Most-recent period margin calculations
        if sorted_income_periods:
            latest = sorted_income_periods[-1]
            income_latest = income_by_period[latest]

            margin_pairs = [
                ("gross", "grossProfit", "revenue", "Gross Margin"),
                ("operating", "operatingIncome", "revenue", "Operating Margin"),
                ("net", "netIncome", "revenue", "Net Margin"),
                ("ebitda", "ebitda", "revenue", "EBITDA Margin"),
                ("rd", "researchAndDevelopmentExpenses", "revenue", "R&D Intensity"),
            ]
            for suffix, num_fld, den_fld, label in margin_pairs:
                num_chunk = income_latest.get(num_fld)
                den_chunk = income_latest.get(den_fld)
                if num_chunk and den_chunk and num_chunk.numeric_value is not None and den_chunk.numeric_value is not None:
                    op = "rd_intensity" if suffix == "rd" else "margin"
                    inputs: dict[str, Any]
                    if suffix == "rd":
                        inputs = {
                            "rd_expenses": num_chunk.numeric_value,
                            "revenue": den_chunk.numeric_value,
                        }
                    else:
                        inputs = {
                            "numerator": num_chunk.numeric_value,
                            "denominator": den_chunk.numeric_value,
                        }
                    req = CalculationRequest(
                        operation=op,
                        inputs=inputs,
                        description=f"{ticker} {label} ({latest})",
                        chunk_ids_used=[num_chunk.chunk_id, den_chunk.chunk_id],
                    )
                    calc_requests.append(req)
                    calc_results.append(self.calculator.compute(req))

        # Most-recent balance sheet ratios
        if sorted_balance_periods:
            latest_b = sorted_balance_periods[-1]
            balance_latest = balance_by_period[latest_b]

            ca_chunk = balance_latest.get("totalCurrentAssets")
            cl_chunk = balance_latest.get("totalCurrentLiabilities")
            if ca_chunk and cl_chunk and ca_chunk.numeric_value and cl_chunk.numeric_value:
                req = CalculationRequest(
                    operation="current_ratio",
                    inputs={
                        "current_assets": ca_chunk.numeric_value,
                        "current_liabilities": cl_chunk.numeric_value,
                    },
                    description=f"{ticker} Current Ratio ({latest_b})",
                    chunk_ids_used=[ca_chunk.chunk_id, cl_chunk.chunk_id],
                )
                calc_requests.append(req)
                calc_results.append(self.calculator.compute(req))

            debt_chunk = balance_latest.get("totalDebt")
            eq_chunk = balance_latest.get("totalStockholdersEquity")
            if debt_chunk and eq_chunk and debt_chunk.numeric_value is not None and eq_chunk.numeric_value:
                req = CalculationRequest(
                    operation="debt_to_equity",
                    inputs={
                        "total_debt": debt_chunk.numeric_value,
                        "equity": eq_chunk.numeric_value,
                    },
                    description=f"{ticker} Debt-to-Equity ({latest_b})",
                    chunk_ids_used=[debt_chunk.chunk_id, eq_chunk.chunk_id],
                )
                calc_requests.append(req)
                calc_results.append(self.calculator.compute(req))

            cash_chunk = balance_latest.get("cashAndCashEquivalents")
            if debt_chunk and cash_chunk and debt_chunk.numeric_value is not None and cash_chunk.numeric_value is not None:
                req = CalculationRequest(
                    operation="net_debt",
                    inputs={
                        "total_debt": debt_chunk.numeric_value,
                        "cash": cash_chunk.numeric_value,
                    },
                    description=f"{ticker} Net Debt ({latest_b})",
                    chunk_ids_used=[debt_chunk.chunk_id, cash_chunk.chunk_id],
                )
                calc_requests.append(req)
                calc_results.append(self.calculator.compute(req))

        # FCF margin (most recent period)
        if sorted_cashflow_periods and sorted_income_periods:
            latest_cf = sorted_cashflow_periods[-1]
            cashflow_latest = cashflow_by_period[latest_cf]
            fcf_chunk = cashflow_latest.get("freeCashFlow")
            rev_chunk = income_by_period.get(sorted_income_periods[-1], {}).get("revenue")
            if fcf_chunk and rev_chunk and fcf_chunk.numeric_value is not None and rev_chunk.numeric_value:
                req = CalculationRequest(
                    operation="fcf_margin",
                    inputs={
                        "free_cash_flow": fcf_chunk.numeric_value,
                        "revenue": rev_chunk.numeric_value,
                    },
                    description=f"{ticker} FCF Margin ({latest_cf})",
                    chunk_ids_used=[fcf_chunk.chunk_id, rev_chunk.chunk_id],
                )
                calc_requests.append(req)
                calc_results.append(self.calculator.compute(req))

        logger.info(
            "FinancialExtractor: %d facts, %d calcs for %s",
            len(facts), len(calc_results), ticker,
        )
        return facts, calc_requests, calc_results
