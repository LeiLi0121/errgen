"""
baseline_prediction_agent node

Generates the "Investment Recommendation & Outlook" section without running
the checker / reviser loop. This is the intended ablation baseline for
evaluation: the report is produced directly from the initial section drafts.
"""

from __future__ import annotations

import logging

from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    EvidenceChunk,
    ReportSection,
    SourceType,
    VerificationStatus,
)
from errgen.nodes.prediction_agent import _PredictionAgent

logger = logging.getLogger(__name__)


def _build_draft_section_summary(sections: list[ReportSection]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(
            f"## {section.section_name} "
            f"(status: {section.verification_status.value})"
        )
        for para in section.paragraphs:
            if para.text.strip():
                text = para.text[:300]
                if len(para.text) > 300:
                    text += "..."
                lines.append(text)
        lines.append("")
    return "\n".join(lines)


def baseline_prediction_agent(state: dict) -> dict:
    """
    Generate the investment recommendation section directly from draft sections.

    No checker / reviser loop is executed here. Paragraphs therefore keep the
    default FAIL status to indicate that they are unverified baseline output.
    """
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    ticker: str = state["ticker"]
    as_of: str | None = state.get("as_of_date")

    section_summary = _build_draft_section_summary(sections)
    key_chunks: list[EvidenceChunk] = (
        [c for c in all_chunks if c.source_type != SourceType.NEWS][:25]
        + [c for c in all_chunks if c.source_type == SourceType.NEWS][:10]
    )

    agent = _PredictionAgent()
    paragraphs: list[AnalysisParagraph] = agent.generate(
        ticker=ticker,
        request_context=section_summary,
        chunks=key_chunks,
        calc_results=calc_results,
        as_of_date=as_of,
    )

    pred_section = ReportSection(
        section_name="Investment Recommendation & Outlook",
        section_order=len(sections) + 1,
        paragraphs=paragraphs,
        verification_status=VerificationStatus.FAIL,
    )

    logger.info(
        "baseline_prediction_agent: generated %d unverified paragraphs",
        len(paragraphs),
    )

    return {
        "analysis_sections": list(sections) + [pred_section],
        "paragraph_drafts": list(paragraphs),
        "has_blocking_issues": False,
    }
