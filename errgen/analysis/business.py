"""Business / competitive positioning analysis agent."""

from __future__ import annotations

from errgen.analysis.base import BaseAnalysisAgent


class BusinessAnalysisAgent(BaseAnalysisAgent):
    """
    Writes the 'Business & Competitive Analysis' section.

    Focus areas:
    - Business model and revenue streams
    - Competitive positioning and moat
    - Key product/service areas and their strategic importance
    - Management and corporate governance signals
    - Strategic initiatives and M&A activity
    """

    section_name = "Business & Competitive Analysis"

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

Write the 'Business & Competitive Analysis' section of a professional equity research report.

Cover:
1. Core business model: how the company generates revenue (product segments, geography)
2. Competitive positioning: market share, moat, differentiation vs peers
3. Key growth drivers: which products, markets, or initiatives are driving growth
4. Strategic initiatives: partnerships, acquisitions, new market entries
5. Management quality signals: commentary from filings, news, analyst sources

Ground every statement in the provided evidence.
Cite every claim with chunk_ids. If the evidence does not cover a sub-topic,
note that explicitly rather than speculating.

=== EVIDENCE CHUNKS ===
{chunk_block}

=== CALCULATION RESULTS ===
{calc_block}

Return the JSON response as specified."""
