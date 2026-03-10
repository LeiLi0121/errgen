"""Financial analysis agent – revenue, margins, growth, cash flow, balance sheet."""

from __future__ import annotations

from errgen.analysis.base import BaseAnalysisAgent


class FinancialAnalysisAgent(BaseAnalysisAgent):
    """
    Writes the 'Financial Analysis' section of the research report.

    Focus areas:
    - Revenue trends and YoY growth
    - Profitability (gross / operating / net margins)
    - EBITDA and cash generation
    - Balance sheet strength (liquidity, leverage)
    - Free cash flow quality
    - R&D investment intensity
    """

    section_name = "Financial Analysis"

    def _build_user_prompt(
        self,
        ticker: str,
        request_context: str,
        chunk_block: str,
        calc_block: str,
        as_of_date: str | None,
    ) -> str:
        return f"""Ticker: {ticker}
Report context: {request_context}
As-of date: {as_of_date or "not specified"}

Write the 'Financial Analysis' section of a professional equity research report.

Cover ALL of the following sub-topics (one or two paragraphs each):
1. Revenue trends and YoY growth rates (use the YoY growth table calc if available)
2. Profitability: gross margin, operating margin, net margin trends
3. EBITDA and cash flow quality (operating cash flow, free cash flow margin)
4. Balance sheet: liquidity (current ratio), leverage (debt-to-equity, net debt)
5. R&D investment intensity relative to revenue

Use ONLY the evidence chunks and calculation results provided below.
Cite every numerical claim with its chunk_id or calc_id.

=== EVIDENCE CHUNKS ===
{chunk_block}

=== CALCULATION RESULTS ===
{calc_block}

Return the JSON response as specified."""
