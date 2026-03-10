"""
Reviser agent.

Given a paragraph that failed the checker, and the list of checker issues,
this agent produces a corrected version of the paragraph that:
  - addresses all critical and major issues
  - remains grounded in the original evidence chunks
  - preserves valid content that had no issues

The reviser returns an updated AnalysisParagraph (same paragraph_id, new text
and potentially updated chunk_ids / calc_ids).
"""

from __future__ import annotations

import logging
from copy import deepcopy

from errgen.config import Config
from errgen.llm import build_messages, chat_json
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    Citation,
    CheckerIssue,
    CheckerVerdict,
    EvidenceChunk,
    RevisionRecord,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_REVISER_SYSTEM = """You are a senior equity research editor revising a paragraph
based on structured fact-checker feedback.

Your task:
1. Address EVERY critical and major issue identified by the checker.
2. Keep valid, well-supported content unchanged where possible.
3. Maintain professional research report tone.
4. Every factual claim in the revised paragraph must be supported by a cited chunk_id.
5. Every number must come from a cited chunk or calc_id.
6. Do NOT introduce new claims not in the evidence.
7. If a claim cannot be supported by any available evidence, remove it entirely.

Return a JSON object:
{
  "text": "The revised paragraph text.",
  "chunk_ids": ["id1", "id2"],
  "calc_ids": ["calc_id1"],
  "changes_summary": "Brief description of what was changed and why."
}
"""


class ReviserAgent:
    """
    Revises a failed paragraph using checker feedback and original evidence.
    """

    def revise(
        self,
        paragraph: AnalysisParagraph,
        verdict: CheckerVerdict,
        chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        iteration: int,
    ) -> tuple[AnalysisParagraph, RevisionRecord]:
        """
        Produce a revised AnalysisParagraph addressing the checker's issues.

        Returns:
            revised_paragraph: a new AnalysisParagraph with updated text/citations
            revision_record:   audit record of what was changed
        """
        blocking_issues = [
            i for i in verdict.issues
            if i.severity.value in ("critical", "major")
        ]
        if not blocking_issues:
            logger.warning(
                "ReviserAgent: called with no blocking issues for paragraph %s; "
                "returning paragraph unchanged.",
                paragraph.paragraph_id,
            )
            record = RevisionRecord(
                paragraph_id=paragraph.paragraph_id,
                iteration=iteration,
                original_text=paragraph.text,
                revised_text=paragraph.text,
                issues_addressed=[],
                changes_summary="No blocking issues – no changes made.",
            )
            return paragraph, record

        issues_block = self._format_issues(blocking_issues)
        chunk_block = self._format_chunks(chunks)
        calc_block = self._format_calcs(calc_results)

        user_prompt = (
            f"=== ORIGINAL PARAGRAPH (Section: {paragraph.section_name}) ===\n"
            f"{paragraph.text}\n\n"
            f"=== CHECKER ISSUES TO FIX ===\n{issues_block}\n\n"
            f"=== ALL AVAILABLE EVIDENCE CHUNKS ===\n{chunk_block}\n\n"
            f"=== CALCULATION RESULTS ===\n{calc_block}\n\n"
            f"Current cited chunk IDs: {paragraph.chunk_ids}\n"
            f"Current cited calc IDs:  {paragraph.calc_ids}\n\n"
            f"Revise the paragraph to fix all critical and major issues. "
            f"Return the JSON as specified."
        )

        messages = build_messages(_REVISER_SYSTEM, user_prompt)

        try:
            raw = chat_json(messages, model=Config.OPENAI_MODEL)
        except Exception as exc:
            logger.error(
                "ReviserAgent LLM call failed for paragraph %s: %s",
                paragraph.paragraph_id, exc,
            )
            # Return original paragraph unchanged; pipeline will mark as unresolved
            record = RevisionRecord(
                paragraph_id=paragraph.paragraph_id,
                iteration=iteration,
                original_text=paragraph.text,
                revised_text=paragraph.text,
                issues_addressed=[],
                changes_summary=f"Revision failed: {exc}",
            )
            return paragraph, record

        revised_text = (raw.get("text") or "").strip()
        if not revised_text:
            revised_text = paragraph.text
            logger.warning(
                "ReviserAgent: empty revision text for %s; keeping original.",
                paragraph.paragraph_id,
            )

        # Validate cited IDs
        available_chunk_ids = {c.chunk_id for c in chunks}
        available_calc_ids = {c.calc_id for c in calc_results}
        chunk_map = {c.chunk_id: c for c in chunks}

        new_chunk_ids = [
            cid for cid in raw.get("chunk_ids", [])
            if cid in available_chunk_ids
        ]
        new_calc_ids = [
            cid for cid in raw.get("calc_ids", [])
            if cid in available_calc_ids
        ]

        # Rebuild citations
        new_citations: list[Citation] = []
        for cid in new_chunk_ids:
            chunk = chunk_map.get(cid)
            if chunk:
                new_citations.append(
                    Citation(
                        chunk_id=cid,
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        text_snippet=chunk.text[:300],
                    )
                )

        # Build revised paragraph (copy, then update mutable fields)
        revised_para = deepcopy(paragraph)
        revised_para.text = revised_text
        revised_para.chunk_ids = new_chunk_ids
        revised_para.calc_ids = new_calc_ids
        revised_para.citations = new_citations
        revised_para.iteration = iteration
        revised_para.verification_status = VerificationStatus.FAIL  # checker will re-evaluate

        record = RevisionRecord(
            paragraph_id=paragraph.paragraph_id,
            iteration=iteration,
            original_text=paragraph.text,
            revised_text=revised_text,
            issues_addressed=[i.issue_id for i in blocking_issues],
            changes_summary=raw.get("changes_summary", ""),
        )

        logger.info(
            "ReviserAgent: revised paragraph %s (iteration %d, %d issues addressed)",
            paragraph.paragraph_id, iteration, len(blocking_issues),
        )
        return revised_para, record

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_issues(issues: list[CheckerIssue]) -> str:
        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(
                f"Issue {i}: [{issue.severity.value.upper()}] {issue.issue_type.value}\n"
                f"  Offending: {issue.offending_span or 'N/A'}\n"
                f"  Explanation: {issue.explanation}\n"
                f"  Relevant chunks: {issue.relevant_chunk_ids}\n"
                f"  Fix direction: {issue.recommended_fix}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_chunks(chunks: list[EvidenceChunk], max_chars: int = 10000) -> str:
        lines = []
        total = 0
        for chunk in chunks:
            entry = (
                f"[CHUNK_ID: {chunk.chunk_id}]\n"
                f"Type: {chunk.source_type.value} | Period: {chunk.period or 'N/A'}\n"
                f"{chunk.text[:500]}\n---\n"
            )
            if total + len(entry) > max_chars:
                lines.append("[...chunks truncated...]")
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)

    @staticmethod
    def _format_calcs(calcs: list[CalculationResult]) -> str:
        if not calcs:
            return "(none)"
        lines = []
        for calc in calcs:
            if calc.error:
                lines.append(f"[CALC_ID: {calc.calc_id}] ERROR: {calc.error}")
            else:
                result_str = (
                    f"{calc.result:.4f}"
                    if isinstance(calc.result, float)
                    else str(calc.result)
                )
                lines.append(
                    f"[CALC_ID: {calc.calc_id}] {calc.description} → {result_str}"
                )
        return "\n".join(lines)
