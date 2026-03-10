"""
Checker agent.

Validates a single AnalysisParagraph against its cited evidence chunks and
calculation results. Returns a structured CheckerVerdict with individual
CheckerIssue objects for every detected problem.

Issue types checked
-------------------
  unsupported_claim  – a factual claim has no supporting chunk
  citation_mismatch  – cited chunk does not actually support the specific claim
  hallucination      – content invented beyond what evidence allows
  numerical_error    – wrong number, wrong formula result
  internal_inconsistency – contradicts previously validated content
  scope_violation    – references data beyond the as-of date
  overclaiming       – uses language implying certainty beyond what evidence shows

The checker returns "pass" only when there are ZERO critical or major issues.
Minor issues (style, phrasing) do not block passage.
"""

from __future__ import annotations

import json
import logging

from errgen.config import Config
from errgen.llm import build_messages, chat_json
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    CheckerIssue,
    CheckerVerdict,
    EvidenceChunk,
    IssueSeverity,
    IssueType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_CHECKER_SYSTEM = """You are a rigorous fact-checker for professional equity research reports.

Your task is to verify that a given paragraph is accurately supported by the
evidence chunks and calculation results it cites.

Check for ALL of the following issue types:

1. unsupported_claim   – A factual claim (number, event, company statement) is 
                         not found in any cited chunk.
2. citation_mismatch   – A chunk is cited but its content does NOT support the
                         specific claim being made.
3. hallucination       – Content that is invented, inferred beyond what the 
                         evidence allows, or not traceable to any provided source.
4. numerical_error     – A number in the text does not match the cited chunk or 
                         calculation result (wrong value, wrong formula, wrong unit).
5. internal_inconsistency – The paragraph contradicts a previously verified fact
                         (provided in the 'Verified context' section).
6. scope_violation     – The paragraph references events, data, or conditions 
                         after the stated as-of date.
7. overclaiming        – Language implies certainty that the evidence does not 
                         warrant (e.g. "will", "definitely", "guaranteed" for 
                         uncertain outcomes).

Return a JSON object:
{
  "status": "pass" | "fail",
  "issues": [
    {
      "issue_type": "<one of the 7 types above>",
      "severity": "critical | major | minor",
      "offending_span": "<the exact problematic text from the paragraph, or null>",
      "explanation": "<why this is an issue>",
      "relevant_chunk_ids": ["id1", ...],
      "relevant_calc_ids": ["calc_id1", ...],
      "recommended_fix": "<specific actionable guidance for revision>"
    }
  ]
}

Severity guide:
  critical – factual error, fabricated number, wrong calculation
  major    – claim not supported by cited evidence, significant citation mismatch
  minor    – style issue, mild overclaiming, imprecise language

Set status to "pass" if and only if there are ZERO critical or major issues.
An empty issues list also means "pass".
Return an empty issues list if the paragraph is fully supported and correct.
"""


class CheckerAgent:
    """
    Checks a single AnalysisParagraph against evidence and calculation results.

    Usage:
        checker = CheckerAgent()
        verdict = checker.check(paragraph, chunks, calc_results, iteration=0)
    """

    def check(
        self,
        paragraph: AnalysisParagraph,
        chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        iteration: int = 0,
        as_of_date: str | None = None,
        verified_context: str = "",
    ) -> CheckerVerdict:
        """
        Validate a paragraph against its cited evidence.

        Parameters
        ----------
        paragraph:        The paragraph to check.
        chunks:           ALL available evidence chunks for this section
                          (not just those cited – checker can detect missing cites).
        calc_results:     ALL available calculation results.
        iteration:        Current revision iteration number.
        as_of_date:       Constraint date for scope_violation checks.
        verified_context: Summary of previously verified facts for consistency
                          checking.

        Returns
        -------
        CheckerVerdict with status=PASS or FAIL and a list of CheckerIssue.
        """
        # Build evidence block (only the chunks cited by this paragraph, plus a
        # summary of all available chunks to detect missing citations)
        cited_chunks = [c for c in chunks if c.chunk_id in paragraph.chunk_ids]
        uncited_chunks = [c for c in chunks if c.chunk_id not in paragraph.chunk_ids]

        cited_block = self._format_chunks(cited_chunks, label="CITED CHUNKS")
        available_ids_note = (
            f"\nOther available (but uncited) chunk IDs: "
            f"{[c.chunk_id for c in uncited_chunks[:20]]}"
            if uncited_chunks
            else ""
        )

        cited_calcs = [c for c in calc_results if c.calc_id in paragraph.calc_ids]
        calc_block = self._format_calcs(cited_calcs)

        user_prompt = (
            f"=== PARAGRAPH TO CHECK ===\n"
            f"Paragraph ID: {paragraph.paragraph_id}\n"
            f"Section: {paragraph.section_name}\n\n"
            f"{paragraph.text}\n\n"
            f"Cited chunk IDs: {paragraph.chunk_ids}\n"
            f"Cited calc IDs:  {paragraph.calc_ids}\n\n"
            f"=== CITED EVIDENCE CHUNKS ===\n{cited_block}{available_ids_note}\n\n"
            f"=== CITED CALCULATION RESULTS ===\n{calc_block}\n\n"
            + (f"=== AS-OF DATE ===\n{as_of_date}\n\n" if as_of_date else "")
            + (
                f"=== PREVIOUSLY VERIFIED FACTS (for consistency check) ===\n"
                f"{verified_context}\n\n"
                if verified_context
                else ""
            )
            + "Check this paragraph for all issue types and return the JSON verdict."
        )

        messages = build_messages(_CHECKER_SYSTEM, user_prompt)

        try:
            raw = chat_json(messages, model=Config.OPENAI_MODEL)
        except Exception as exc:
            logger.error(
                "CheckerAgent LLM call failed for paragraph %s: %s",
                paragraph.paragraph_id, exc,
            )
            # If the checker itself fails, we cannot validate – mark as unresolved
            return CheckerVerdict(
                paragraph_id=paragraph.paragraph_id,
                status=VerificationStatus.UNRESOLVED,
                issues=[
                    CheckerIssue(
                        issue_type=IssueType.UNSUPPORTED_CLAIM,
                        severity=IssueSeverity.CRITICAL,
                        paragraph_id=paragraph.paragraph_id,
                        explanation=f"Checker LLM call failed: {exc}",
                        recommended_fix="Re-run the checker.",
                    )
                ],
                iteration=iteration,
            )

        return self._parse_verdict(raw, paragraph.paragraph_id, iteration)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunks(chunks: list[EvidenceChunk], label: str = "CHUNKS") -> str:
        if not chunks:
            return f"({label}: none cited)\n"
        lines = []
        for chunk in chunks:
            lines.append(
                f"[CHUNK_ID: {chunk.chunk_id}]\n"
                f"Type: {chunk.source_type.value} | Period: {chunk.period or 'N/A'} | "
                f"Field: {chunk.field_name or 'N/A'}\n"
                f"{chunk.text[:600]}\n---"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_calcs(calcs: list[CalculationResult]) -> str:
        if not calcs:
            return "(no calculations cited)\n"
        lines = []
        for calc in calcs:
            if calc.error:
                lines.append(
                    f"[CALC_ID: {calc.calc_id}] ERROR: {calc.error}\n---"
                )
            else:
                result_str = (
                    f"{calc.result:.4f}"
                    if isinstance(calc.result, float)
                    else str(calc.result)
                )
                lines.append(
                    f"[CALC_ID: {calc.calc_id}]\n"
                    f"Description: {calc.description}\n"
                    f"Formula: {calc.formula_description}\n"
                    f"Result: {result_str}\n---"
                )
        return "\n".join(lines)

    @staticmethod
    def _parse_verdict(
        raw: dict, paragraph_id: str, iteration: int
    ) -> CheckerVerdict:
        """Parse the LLM JSON response into a CheckerVerdict."""
        status_str = raw.get("status", "fail").lower()
        status = (
            VerificationStatus.PASS
            if status_str == "pass"
            else VerificationStatus.FAIL
        )

        issues: list[CheckerIssue] = []
        for item in raw.get("issues", []):
            try:
                issue_type = IssueType(item.get("issue_type", "unsupported_claim"))
            except ValueError:
                issue_type = IssueType.UNSUPPORTED_CLAIM

            try:
                severity = IssueSeverity(item.get("severity", "major"))
            except ValueError:
                severity = IssueSeverity.MAJOR

            issues.append(
                CheckerIssue(
                    issue_type=issue_type,
                    severity=severity,
                    paragraph_id=paragraph_id,
                    offending_span=item.get("offending_span"),
                    explanation=item.get("explanation", ""),
                    relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                    relevant_calc_ids=item.get("relevant_calc_ids", []),
                    recommended_fix=item.get("recommended_fix", ""),
                )
            )

        # Safety check: if there are critical/major issues but status says pass,
        # override to fail (the LLM may have been inconsistent)
        has_blocking = any(
            i.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)
            for i in issues
        )
        if has_blocking and status == VerificationStatus.PASS:
            logger.warning(
                "CheckerAgent: LLM said pass but found blocking issues for %s. "
                "Overriding to fail.",
                paragraph_id,
            )
            status = VerificationStatus.FAIL

        logger.info(
            "CheckerAgent: paragraph %s → %s (%d issues: %d critical/major)",
            paragraph_id,
            status.value,
            len(issues),
            sum(1 for i in issues if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)),
        )

        return CheckerVerdict(
            paragraph_id=paragraph_id,
            status=status,
            issues=issues,
            iteration=iteration,
        )
