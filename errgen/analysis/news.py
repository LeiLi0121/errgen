"""News / recent developments analysis agent."""

from __future__ import annotations

from errgen.analysis.base import BaseAnalysisAgent


class NewsAnalysisAgent(BaseAnalysisAgent):
    """
    Writes the 'Recent Developments' section of the research report.

    Focus areas:
    - Key product launches, partnerships, acquisitions
    - Earnings announcements and guidance
    - Analyst rating changes
    - Regulatory or legal developments
    - Macro / geopolitical news affecting the company
    - Market sentiment and investor reactions
    """

    section_name = "Recent Developments"

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

Write the 'Recent Developments' section of a professional equity research report.
This section should synthesise the most important news events and strategic
developments up to the as-of date.

Cover:
1. Major product announcements, partnerships, or acquisitions
2. Financial results releases, guidance changes, or management commentary
3. Analyst / rating agency actions
4. Regulatory, legal, or policy developments that affect the company
5. Competitive dynamics or notable market events

Prioritise high-impact events. Be chronological where useful.
Cite every claim with the relevant chunk_id(s).

=== EVIDENCE CHUNKS (News) ===
{chunk_block}

=== CALCULATION RESULTS ===
{calc_block}

Return the JSON response as specified."""
