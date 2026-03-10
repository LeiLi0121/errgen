"""Risk analysis agent."""

from __future__ import annotations

from errgen.analysis.base import BaseAnalysisAgent


class RiskAnalysisAgent(BaseAnalysisAgent):
    """
    Writes the 'Risk Analysis' section of the research report.

    Synthesises risks across:
    - Macro / geopolitical risks
    - Competitive and market share risks
    - Financial risks (leverage, liquidity, revenue concentration)
    - Regulatory and legal risks
    - Operational and supply chain risks
    - Technology disruption risks
    """

    section_name = "Risk Analysis"

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

Write the 'Risk Analysis' section of a professional equity research report.

Identify and discuss material risks across these categories:
1. Macro / geopolitical risks (e.g. export controls, currency, interest rates)
2. Competitive risks (market share threats, new entrants, pricing pressure)
3. Financial risks (leverage, liquidity, concentration, margin compression)
4. Regulatory and legal risks (antitrust, compliance, government action)
5. Operational risks (supply chain, talent, execution)
6. Technology and disruption risks (platform shifts, obsolescence)

For each risk:
- State the risk clearly
- Explain the mechanism by which it affects the company
- Cite supporting evidence from news or financial data
- Indicate severity where possible (high / medium / low based on evidence)

Use ONLY evidence from the provided chunks. Do not invent risks not evidenced.
Cite every claim with the relevant chunk_id(s).

=== EVIDENCE CHUNKS ===
{chunk_block}

=== CALCULATION RESULTS ===
{calc_block}

Return the JSON response as specified."""
