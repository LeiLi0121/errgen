"""
prediction_agent node

Generates the "Investment Recommendation & Outlook" section.

GATE: If any upstream section has unresolved critical/major issues, the
prediction section is skipped and a warning is added to state.  This
prevents the LLM from synthesising an unreliable recommendation on top
of unverified evidence.

The generated paragraphs go through the same inline checker → reviser
loop used by the main sections, ensuring the recommendation is
evidence-grounded before being appended to analysis_sections.
"""

from __future__ import annotations

import logging

from errgen.analysis.base import BaseAnalysisAgent
from errgen.config import Config
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    CheckerVerdict,
    EvidenceChunk,
    IssueSeverity,
    ReportSection,
    RevisionRecord,
    SourceType,
    VerificationStatus,
)
from errgen.verification import CheckerAgent, ReviserAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction-specific system prompt (inline – not shared with analysis.base)
# ---------------------------------------------------------------------------

_PREDICTION_SYSTEM = """You are a senior equity research analyst writing the
'Investment Recommendation & Outlook' section of a professional research report.

This is the final section.  It synthesises the verified analysis above into a
forward-looking investment view.

RULES:
1. Base your recommendation ONLY on the verified section summaries and evidence provided.
2. Cite chunk_ids and calc_ids that support every claim.
3. Acknowledge key risks and uncertainties explicitly.
4. Use language appropriate to the evidence certainty level.
5. Do NOT promise returns or guarantee outcomes.
6. State an explicit rating in the FIRST sentence of the section using exactly one
   of these labels: Buy / Hold / Sell.
7. Compare the company to the benchmark signal when benchmark data is provided.
8. If evidence is genuinely balanced or insufficient for a firm recommendation, say so clearly and default to Hold.
9. Do not default to Hold solely because ordinary risks or some volatility exist. When the preponderance of verified evidence supports favorable fundamentals, catalysts, and risk/reward, a Buy is appropriate.
10. Do not overweight single-quarter earnings noise, generic macro uncertainty, or routine market volatility when the broader verified evidence is strong.
11. Remain disciplined: do not force a Buy when the verified evidence is mixed, fragile, or too incomplete to support a constructive view.
12. Do not downgrade to Hold merely because of heavy capex, elevated R&D, temporary
   margin compression, or sequential moderation if the broader verified evidence still
   shows strong fundamentals, credible catalysts, and manageable balance-sheet risk.
13. Base the recommendation on medium-term risk/reward implied by the verified
   evidence, not just the most recent quarter in isolation.
14. Do NOT include chunk refs like C001 or calc refs like K001 in the text itself.
   Put references only in the chunk_ids / calc_ids arrays.
15. Write 2-3 paragraphs only. Do not return 4 or more paragraphs.

Return JSON:
{
  "paragraphs": [
    {
      "text": "...",
      "chunk_ids": ["id1"],
      "calc_ids": ["calc1"]
    }
  ]
}
"""


class _PredictionAgent(BaseAnalysisAgent):
    section_name = "Investment Recommendation & Outlook"

    def _build_system_prompt(self, as_of_date: str | None) -> str:
        return _PREDICTION_SYSTEM

    def _build_user_prompt(
        self,
        ticker: str,
        request_context: str,
        chunk_block: str,
        calc_block: str,
        as_of_date: str | None,
    ) -> str:
        return (
            f"Ticker: {ticker}\n"
            f"Benchmark: {Config.BENCHMARK_TICKER}\n"
            f"As-of date: {as_of_date or 'not specified'}\n\n"
            f"=== VERIFIED SECTION SUMMARIES ===\n{request_context}\n\n"
            f"=== KEY EVIDENCE CHUNKS ===\n{chunk_block}\n\n"
            f"=== CALCULATION RESULTS ===\n{calc_block}\n\n"
            "Write the 'Investment Recommendation & Outlook' section.\n"
            "Use the recent company-vs-benchmark price signal when it is available, "
            "but do not rely on market momentum alone: reconcile it with fundamentals, "
            "filings, recent developments, and risks. Calibrate conservatism: "
            "distinguish ordinary uncertainty from evidence that genuinely weakens the thesis. "
            "Open with a direct rating sentence, then justify it using the strongest "
            "verified bull and bear factors. If the net evidence is constructive, do not "
            "retreat to Hold just because one quarter softened or investment spending is high."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_section_summary(sections: list[ReportSection]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(
            f"## {section.section_name} "
            f"(status: {section.verification_status.value})"
        )
        for para in section.paragraphs:
            if para.verification_status == VerificationStatus.PASS:
                lines.append(
                    para.text[:300] + ("..." if len(para.text) > 300 else "")
                )
        lines.append("")
    return "\n".join(lines)


def _inline_verify(
    paragraphs: list[AnalysisParagraph],
    key_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
    as_of: str | None,
    verified_context: str,
) -> tuple[list[CheckerVerdict], list[RevisionRecord]]:
    """Run the checker → reviser loop inline for the prediction section."""
    checker = CheckerAgent()
    reviser = ReviserAgent()
    max_iter = Config.MAX_REVISION_ITERATIONS
    new_verdicts: list[CheckerVerdict] = []
    new_revisions: list[RevisionRecord] = []

    for para in paragraphs:
        for iteration in range(max_iter + 1):
            verdict = checker.check(
                paragraph=para,
                chunks=key_chunks,
                calc_results=calc_results,
                iteration=iteration,
                as_of_date=as_of,
                verified_context=verified_context,
            )
            para.checker_verdicts.append(verdict)
            new_verdicts.append(verdict)

            if verdict.status == VerificationStatus.PASS:
                para.verification_status = VerificationStatus.PASS
                break

            if iteration >= max_iter:
                para.verification_status = VerificationStatus.UNRESOLVED
                logger.warning(
                    "prediction_agent: paragraph %s UNRESOLVED after %d iters",
                    para.paragraph_id, max_iter,
                )
                break

            revised_para, revision_record = reviser.revise(
                paragraph=para,
                verdict=verdict,
                chunks=key_chunks,
                calc_results=calc_results,
                iteration=iteration + 1,
            )
            para.text = revised_para.text
            para.chunk_ids = revised_para.chunk_ids
            para.calc_ids = revised_para.calc_ids
            para.citations = revised_para.citations
            para.revision_history.append(revision_record)
            new_revisions.append(revision_record)

    return new_verdicts, new_revisions


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def prediction_agent(state: dict) -> dict:
    """Generate and verify the investment recommendation section."""
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    ticker: str = state["ticker"]
    as_of: str | None = state.get("as_of_date")

    # -- Gate: skip if upstream sections have unresolved blocking issues ------
    blocking = [s.section_name for s in sections if s.unresolved_issues]
    if blocking:
        msg = (
            f"Prediction section SKIPPED: the following sections have unresolved "
            f"blocking issues: {blocking}. "
            f"Resolve those issues before generating a recommendation."
        )
        logger.warning(msg)
        return {
            "has_blocking_issues": True,
            "warnings": [msg],
        }

    # -- Generate recommendation section ------------------------------------
    section_summary = _build_section_summary(sections)
    key_chunks: list[EvidenceChunk] = (
        [c for c in all_chunks if c.source_type != SourceType.NEWS][:25]
        + [c for c in all_chunks if c.source_type == SourceType.NEWS][:10]
    )

    agent = _PredictionAgent()
    paragraphs = agent.generate(
        ticker=ticker,
        request_context=section_summary,
        chunks=key_chunks,
        calc_results=calc_results,
        as_of_date=as_of,
    )

    # -- Inline verification loop -------------------------------------------
    new_verdicts, new_revisions = _inline_verify(
        paragraphs=paragraphs,
        key_chunks=key_chunks,
        calc_results=calc_results,
        as_of=as_of,
        verified_context=section_summary,
    )

    # -- Build prediction section -------------------------------------------
    statuses = {p.verification_status for p in paragraphs}
    if all(s == VerificationStatus.PASS for s in statuses):
        section_status = VerificationStatus.PASS
    elif VerificationStatus.UNRESOLVED in statuses:
        section_status = VerificationStatus.UNRESOLVED
    else:
        section_status = VerificationStatus.FAIL

    unresolved_issues = [
        issue
        for para in paragraphs
        for verdict in para.checker_verdicts
        for issue in verdict.issues
        if (
            para.verification_status == VerificationStatus.UNRESOLVED
            and issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)
        )
    ]

    pred_section = ReportSection(
        section_name="Investment Recommendation & Outlook",
        section_order=len(sections) + 1,
        paragraphs=paragraphs,
        verification_status=section_status,
        unresolved_issues=unresolved_issues,
    )

    logger.info(
        "prediction_agent: recommendation section status=%s",
        section_status.value,
    )

    return {
        "analysis_sections": list(sections) + [pred_section],
        "paragraph_drafts": list(paragraphs),
        "checker_verdicts": new_verdicts,
        "revision_records": new_revisions,
        "has_blocking_issues": False,
    }
