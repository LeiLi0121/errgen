"""
verification_agent node  ·  revise_sections node  ·  route_after_verification

Implements the iterative evidence-grounding loop:

    analysis_agent
         │
         ▼
    verification_agent  ──── all PASS/UNRESOLVED ────► prediction_agent
         ▲                        │
         │                        │ any FAIL?
         │                        ▼
         └──────────────── revise_sections
                           (increments revision_count;
                            marks as UNRESOLVED at max iterations)

Termination guarantee:
  revise_sections marks any paragraph still at FAIL as UNRESOLVED once
  revision_count >= MAX_REVISION_ITERATIONS.  verification_agent then
  finds no FAIL paragraphs and the conditional edge exits the loop.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from errgen.nodes.analysis_agent import (
    select_calcs_for_section_name,
    select_chunks_for_section_name,
)
from errgen.config import Config
from errgen.models import (
    CalculationResult,
    CheckerVerdict,
    EvidenceChunk,
    IssueSeverity,
    ReportSection,
    RevisionRecord,
    VerificationStatus,
)
from errgen.verification import CheckerAgent, ReviserAgent

logger = logging.getLogger(__name__)


def _build_verified_context_snapshot(sections: list[ReportSection]) -> str:
    """
    Freeze the verified context at the start of a checker round.

    This preserves deterministic, round-based semantics when checker calls
    are executed concurrently: paragraphs that PASS during the current round
    do not affect other paragraphs until the next verification round.
    """
    parts: list[str] = []
    for section in sections:
        for para in section.paragraphs:
            if para.verification_status == VerificationStatus.PASS:
                parts.append(f"[{para.section_name}] {para.text[:200]}")
    return "\n".join(parts)


def _run_checker_task(
    paragraph,
    chunks,
    calc_results,
    iteration,
    as_of_date,
    verified_context,
):
    checker = CheckerAgent()
    return checker.check(
        paragraph=paragraph,
        chunks=chunks,
        calc_results=calc_results,
        iteration=iteration,
        as_of_date=as_of_date,
        verified_context=verified_context,
    )


def _run_revision_task(
    paragraph,
    verdict,
    chunks,
    calc_results,
    iteration,
):
    reviser = ReviserAgent()
    return reviser.revise(
        paragraph=paragraph,
        verdict=verdict,
        chunks=chunks,
        calc_results=calc_results,
        iteration=iteration,
    )


# ---------------------------------------------------------------------------
# verification_agent
# ---------------------------------------------------------------------------

def verification_agent(state: dict) -> dict:
    """
    Run CheckerAgent on every paragraph that is currently at FAIL status.

    Paragraphs already at PASS or UNRESOLVED are skipped.
    The set of PASS paragraphs is snapshotted at the start of the round so
    all FAIL paragraphs are checked against the same verified context.
    Returns the updated sections list (overwrite) and new checker verdicts
    (appended).
    """
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    as_of: str | None = state.get("as_of_date")
    revision_count: int = state.get("revision_count", 0)

    new_verdicts: list[CheckerVerdict] = []
    verified_context_snapshot = _build_verified_context_snapshot(sections)
    work_items: list[tuple[int, int, object]] = []

    for section_idx, section in enumerate(sections):
        for para_idx, para in enumerate(section.paragraphs):
            if para.verification_status == VerificationStatus.FAIL:
                work_items.append((section_idx, para_idx, para))

    if work_items:
        max_workers = max(1, min(len(work_items), Config.CHECKER_MAX_CONCURRENCY))
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for section_idx, para_idx, para in work_items:
                future = executor.submit(
                    _run_checker_task,
                    para,
                    all_chunks,
                    calc_results,
                    revision_count,
                    as_of,
                    verified_context_snapshot,
                )
                future_map[future] = (section_idx, para_idx, para)

            results: dict[tuple[int, int], CheckerVerdict] = {}
            for future in as_completed(future_map):
                section_idx, para_idx, para = future_map[future]
                try:
                    results[(section_idx, para_idx)] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception(
                        "verification_agent: unexpected checker failure for paragraph %s: %s",
                        para.paragraph_id,
                        exc,
                    )
                    checker = CheckerAgent()
                    results[(section_idx, para_idx)] = checker.check(
                        paragraph=para,
                        chunks=all_chunks,
                        calc_results=calc_results,
                        iteration=revision_count,
                        as_of_date=as_of,
                        verified_context=verified_context_snapshot,
                    )

        for section_idx, para_idx, para in work_items:
            verdict = results[(section_idx, para_idx)]
            para.checker_verdicts.append(verdict)
            new_verdicts.append(verdict)

            if verdict.status == VerificationStatus.PASS:
                para.verification_status = VerificationStatus.PASS
                logger.info(
                    "verification_agent: paragraph %s PASSED (iter %d)",
                    para.paragraph_id, revision_count,
                )
            else:
                logger.info(
                    "verification_agent: paragraph %s FAILED "
                    "(iter %d, %d issues)",
                    para.paragraph_id, revision_count, len(verdict.issues),
                )

    # Recompute section-level statuses
    for section in sections:
        _update_section_status(section)

    return {
        "analysis_sections": sections,
        "checker_verdicts": new_verdicts,
    }


# ---------------------------------------------------------------------------
# revise_sections
# ---------------------------------------------------------------------------

def revise_sections(state: dict) -> dict:
    """
    Revise every paragraph still at FAIL status.

    If revision_count >= MAX_REVISION_ITERATIONS the paragraph is marked
    UNRESOLVED instead of being revised, guaranteeing loop termination.
    Increments revision_count.
    """
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    revision_count: int = state.get("revision_count", 0)

    new_revisions: list[RevisionRecord] = []
    max_iter = Config.MAX_REVISION_ITERATIONS
    work_items: list[
        tuple[int, int, object, object, list[EvidenceChunk], list[CalculationResult]]
    ] = []

    for section_idx, section in enumerate(sections):
        for para_idx, para in enumerate(section.paragraphs):
            if para.verification_status != VerificationStatus.FAIL:
                continue

            if revision_count >= max_iter:
                para.verification_status = VerificationStatus.UNRESOLVED
                logger.warning(
                    "revise_sections: paragraph %s marked UNRESOLVED "
                    "after %d iterations",
                    para.paragraph_id, revision_count,
                )
                continue

            latest_verdict = (
                para.checker_verdicts[-1] if para.checker_verdicts else None
            )
            if not latest_verdict:
                continue

            relevant_chunks = select_chunks_for_section_name(
                para.section_name,
                all_chunks,
            )
            relevant_calcs = select_calcs_for_section_name(
                para.section_name,
                calc_results,
            )
            work_items.append(
                (
                    section_idx,
                    para_idx,
                    para,
                    latest_verdict,
                    relevant_chunks,
                    relevant_calcs,
                )
            )

    if work_items:
        max_workers = max(1, min(len(work_items), Config.REVISION_MAX_CONCURRENCY))
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for (
                section_idx,
                para_idx,
                para,
                latest_verdict,
                relevant_chunks,
                relevant_calcs,
            ) in work_items:
                future = executor.submit(
                    _run_revision_task,
                    para,
                    latest_verdict,
                    relevant_chunks,
                    relevant_calcs,
                    revision_count + 1,
                )
                future_map[future] = (
                    section_idx,
                    para_idx,
                    para,
                )

            results: dict[tuple[int, int], tuple[object, RevisionRecord]] = {}
            for future in as_completed(future_map):
                section_idx, para_idx, para = future_map[future]
                try:
                    results[(section_idx, para_idx)] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception(
                        "revise_sections: unexpected revision failure for paragraph %s: %s",
                        para.paragraph_id,
                        exc,
                    )
                    record = RevisionRecord(
                        paragraph_id=para.paragraph_id,
                        iteration=revision_count + 1,
                        original_text=para.text,
                        revised_text=para.text,
                        issues_addressed=[],
                        changes_summary=f"Revision task failed unexpectedly: {exc}",
                    )
                    results[(section_idx, para_idx)] = (para, record)

        for section_idx, para_idx, para, _, _, _ in work_items:
            revised_para, revision_record = results[(section_idx, para_idx)]

            # Replace the paragraph object in the section list
            revised_para.revision_history = (
                list(para.revision_history) + [revision_record]
            )
            revised_para.checker_verdicts = list(para.checker_verdicts)
            sections[section_idx].paragraphs[para_idx] = revised_para
            new_revisions.append(revision_record)

            logger.info(
                "revise_sections: revised paragraph %s (iter %d → %d)",
                para.paragraph_id, revision_count, revision_count + 1,
            )

    # Recompute section statuses after revisions
    for section in sections:
        _update_section_status(section)

    return {
        "analysis_sections": sections,
        "revision_records": new_revisions,
        "revision_count": revision_count + 1,
    }


# ---------------------------------------------------------------------------
# Conditional edge function
# ---------------------------------------------------------------------------

def route_after_verification(state: dict) -> str:
    """
    Routing function called after verification_agent.

    Returns:
      "revise_sections"   – if any paragraph is still at FAIL
      "prediction_agent"  – if all paragraphs are PASS or UNRESOLVED
    """
    for section in state["analysis_sections"]:
        for para in section.paragraphs:
            if para.verification_status == VerificationStatus.FAIL:
                return "revise_sections"
    return "prediction_agent"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _update_section_status(section: ReportSection) -> None:
    """Recompute section.verification_status from its paragraphs."""
    if not section.paragraphs:
        return

    statuses = {p.verification_status for p in section.paragraphs}

    if all(s == VerificationStatus.PASS for s in statuses):
        section.verification_status = VerificationStatus.PASS
        section.unresolved_issues = []
    elif VerificationStatus.UNRESOLVED in statuses:
        section.verification_status = VerificationStatus.UNRESOLVED
        section.unresolved_issues = [
            issue
            for para in section.paragraphs
            for verdict in para.checker_verdicts
            for issue in verdict.issues
            if (
                para.verification_status == VerificationStatus.UNRESOLVED
                and issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)
            )
        ]
    else:
        section.verification_status = VerificationStatus.FAIL
