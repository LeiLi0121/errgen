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
from errgen.prompt_aliases import (
    aliases_for_ids,
    build_prompt_alias_maps,
    extract_aliases_from_text,
    ids_for_aliases,
    strip_inline_aliases,
    unique_preserve_order,
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
8. Do NOT include chunk refs like C001 or calc refs like K001 in the text itself.
   Put references only in the chunk_ids / calc_ids arrays.

Return a JSON object:
{
  "text": "The revised paragraph text.",
  "chunk_ids": ["C001", "C002"],
  "calc_ids": ["K001"],
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

        alias_maps = build_prompt_alias_maps(chunks, calc_results)
        issues_block = self._format_issues(
            blocking_issues,
            alias_maps.chunk_id_to_alias,
            alias_maps.calc_id_to_alias,
        )
        chunk_block = self._format_chunks(chunks, alias_maps.chunk_id_to_alias)
        calc_block = self._format_calcs(calc_results, alias_maps.calc_id_to_alias)

        user_prompt = (
            f"=== ORIGINAL PARAGRAPH (Section: {paragraph.section_name}) ===\n"
            f"{paragraph.text}\n\n"
            f"=== CHECKER ISSUES TO FIX ===\n{issues_block}\n\n"
            f"=== ALL AVAILABLE EVIDENCE CHUNKS ===\n{chunk_block}\n\n"
            f"=== CALCULATION RESULTS ===\n{calc_block}\n\n"
            f"Current cited chunk refs: {aliases_for_ids(paragraph.chunk_ids, alias_maps.chunk_id_to_alias)}\n"
            f"Current cited calc refs:  {aliases_for_ids(paragraph.calc_ids, alias_maps.calc_id_to_alias)}\n\n"
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
        text_chunk_aliases, text_calc_aliases = extract_aliases_from_text(revised_text)
        cleaned_text = strip_inline_aliases(revised_text) or revised_text

        # Validate cited IDs
        chunk_map = {c.chunk_id: c for c in chunks}

        new_chunk_ids = ids_for_aliases(
            unique_preserve_order(list(raw.get("chunk_ids", [])) + text_chunk_aliases),
            alias_maps.chunk_alias_to_id,
        )
        new_calc_ids = ids_for_aliases(
            unique_preserve_order(list(raw.get("calc_ids", [])) + text_calc_aliases),
            alias_maps.calc_alias_to_id,
        )

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
        revised_para.text = cleaned_text
        revised_para.chunk_ids = new_chunk_ids
        revised_para.calc_ids = new_calc_ids
        revised_para.citations = new_citations
        revised_para.iteration = iteration
        revised_para.verification_status = VerificationStatus.FAIL  # checker will re-evaluate

        record = RevisionRecord(
            paragraph_id=paragraph.paragraph_id,
            iteration=iteration,
            original_text=paragraph.text,
            revised_text=cleaned_text,
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
    def _format_issues(
        issues: list[CheckerIssue],
        chunk_id_to_alias: dict[str, str],
        calc_id_to_alias: dict[str, str],
    ) -> str:
        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(
                f"Issue {i}: [{issue.severity.value.upper()}] {issue.issue_type.value}\n"
                f"  Offending: {issue.offending_span or 'N/A'}\n"
                f"  Explanation: {issue.explanation}\n"
                f"  Relevant chunks: {aliases_for_ids(issue.relevant_chunk_ids, chunk_id_to_alias)}\n"
                f"  Relevant calcs: {aliases_for_ids(issue.relevant_calc_ids, calc_id_to_alias)}\n"
                f"  Fix direction: {issue.recommended_fix}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_chunks(
        chunks: list[EvidenceChunk],
        chunk_id_to_alias: dict[str, str],
        max_chars: int = 10000,
    ) -> str:
        lines = []
        total = 0
        for chunk in chunks:
            entry = (
                f"[CHUNK_REF: {chunk_id_to_alias.get(chunk.chunk_id, chunk.chunk_id)}]\n"
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
    def _format_calcs(
        calcs: list[CalculationResult],
        calc_id_to_alias: dict[str, str],
    ) -> str:
        if not calcs:
            return "(none)"
        lines = []
        for calc in calcs:
            alias = calc_id_to_alias.get(calc.calc_id, calc.calc_id)
            if calc.error:
                lines.append(f"[CALC_REF: {alias}] ERROR: {calc.error}")
            else:
                result_str = (
                    f"{calc.result:.4f}"
                    if isinstance(calc.result, float)
                    else str(calc.result)
                )
                lines.append(
                    f"[CALC_REF: {alias}] {calc.description} → {result_str}"
                )
        return "\n".join(lines)
