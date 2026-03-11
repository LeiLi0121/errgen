"""
report_writer node  (terminal node)

Assembles the FinalReport from verified sections, builds the evidence and
calculation appendices, renders the report to Markdown, and persists the
complete run artifact to disk.

This is the last node in the graph.  It writes `report` and `report_md`
to the state, which the caller can then read from the final state dict.
"""

from __future__ import annotations

import logging
from datetime import datetime

from errgen.models import (
    CalculationResult,
    EvidenceChunk,
    FinalReport,
    ReportSection,
    RunArtifact,
    UserRequest,
)
from errgen.report import ReportAssembler, ReportRenderer
from errgen.run_record import RunRecord

logger = logging.getLogger(__name__)


def report_writer(state: dict) -> dict:
    """Assemble, render, and persist the final equity research report."""
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    warnings: list[str] = state.get("warnings") or []
    run_id: str = state["run_id"]

    # Reconstruct a UserRequest for report metadata
    request = UserRequest(
        raw_text=state.get("query", ""),
        ticker=state["ticker"],
        company_name=state.get("company_name"),
        as_of_date=state.get("as_of_date"),
        focus_areas=state.get("focus") or [],
    )

    # Assemble the FinalReport
    assembler = ReportAssembler()
    report: FinalReport = assembler.assemble(
        request=request,
        sections=sections,
        all_chunks=all_chunks,
        calc_results=calc_results,
        warnings=list(warnings),
    )

    # Render to Markdown
    renderer = ReportRenderer()
    report_md: str = renderer.render(report)

    logger.info(
        "report_writer: report %s | status=%s | %d sections | %d cited chunks",
        report.report_id,
        report.overall_status.value,
        len(sections),
        len(report.evidence_appendix),
    )

    # Persist run artifacts to disk (never raises – logs errors instead)
    _persist_run(state, request, report, report_md, run_id)

    return {
        "report": report,
        "report_md": report_md,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _persist_run(
    state: dict,
    request: UserRequest,
    report: FinalReport,
    report_md: str,
    run_id: str,
) -> None:
    """Construct a RunArtifact and save all artifacts to disk."""
    artifact = RunArtifact(
        run_id=run_id,
        request=request,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
        retrieved_sources=state.get("sources") or [],
        evidence_chunks=state.get("evidence_chunks") or [],
        extracted_facts=state.get("extracted_facts") or [],
        calculation_requests=state.get("calculation_requests") or [],
        calculation_results=state.get("calculations") or [],
        paragraph_drafts=state.get("paragraph_drafts") or [],
        checker_verdicts=state.get("checker_verdicts") or [],
        revision_records=state.get("revision_records") or [],
        final_report=report,
        warnings=state.get("warnings") or [],
        errors=state.get("errors") or [],
    )

    record = RunRecord()
    try:
        run_dir = record.save(artifact)
        logger.info("report_writer: artifacts saved to %s", run_dir)
    except Exception as exc:
        logger.error("report_writer: failed to save artifacts: %s", exc)
