from __future__ import annotations

import importlib
import time
from unittest.mock import patch

from errgen.models import (
    AnalysisParagraph,
    CheckerIssue,
    CheckerVerdict,
    EvidenceChunk,
    IssueSeverity,
    IssueType,
    ReportSection,
    RevisionRecord,
    SourceType,
    VerificationStatus,
)
from errgen.nodes.analysis_agent import select_chunks_for_section_name
from errgen.nodes.verification_agent import revise_sections

verification_agent_module = importlib.import_module("errgen.nodes.verification_agent")


def _issue(paragraph_id: str) -> CheckerIssue:
    return CheckerIssue(
        issue_type=IssueType.CITATION_MISMATCH,
        severity=IssueSeverity.MAJOR,
        paragraph_id=paragraph_id,
        explanation="Needs a better supporting citation.",
        recommended_fix="Use evidence from the relevant section only.",
    )


def test_select_chunks_for_recent_developments_only_returns_news() -> None:
    news_chunk = EvidenceChunk(
        source_id="src-news",
        source_type=SourceType.NEWS,
        text="News event.",
    )
    income_chunk = EvidenceChunk(
        source_id="src-income",
        source_type=SourceType.INCOME_STATEMENT,
        text="Revenue: $10.",
        period="Q1 2025",
        field_name="revenue",
        numeric_value=10.0,
    )

    selected = select_chunks_for_section_name(
        "Recent Developments",
        [news_chunk, income_chunk],
    )

    assert [chunk.chunk_id for chunk in selected] == [news_chunk.chunk_id]


def test_revise_sections_passes_section_scoped_chunks_to_reviser() -> None:
    news_chunk = EvidenceChunk(
        source_id="src-news",
        source_type=SourceType.NEWS,
        text="Important news item.",
    )
    income_chunk = EvidenceChunk(
        source_id="src-income",
        source_type=SourceType.INCOME_STATEMENT,
        text="Revenue: $10.",
        period="Q1 2025",
        field_name="revenue",
        numeric_value=10.0,
    )
    paragraph = AnalysisParagraph(
        section_name="Recent Developments",
        text="Draft paragraph.",
        chunk_ids=[news_chunk.chunk_id],
        verification_status=VerificationStatus.FAIL,
    )
    verdict = CheckerVerdict(
        paragraph_id=paragraph.paragraph_id,
        status=VerificationStatus.FAIL,
        issues=[_issue(paragraph.paragraph_id)],
    )
    paragraph.checker_verdicts.append(verdict)
    section = ReportSection(
        section_name="Recent Developments",
        section_order=1,
        paragraphs=[paragraph],
        verification_status=VerificationStatus.FAIL,
    )
    state = {
        "analysis_sections": [section],
        "evidence_chunks": [news_chunk, income_chunk],
        "calculations": [],
        "revision_count": 0,
    }
    captured_chunk_types: list[SourceType] = []

    def _fake_revise(*, paragraph, verdict, chunks, calc_results, iteration):
        del verdict, calc_results
        captured_chunk_types.extend(chunk.source_type for chunk in chunks)
        revision_record = RevisionRecord(
            paragraph_id=paragraph.paragraph_id,
            iteration=iteration,
            original_text=paragraph.text,
            revised_text=paragraph.text,
            changes_summary="No-op test revision.",
        )
        return paragraph, revision_record

    with patch.object(
        verification_agent_module.ReviserAgent,
        "revise",
        side_effect=_fake_revise,
    ):
        updated = revise_sections(state)

    assert captured_chunk_types == [SourceType.NEWS]
    assert updated["revision_count"] == 1


def test_revise_sections_preserves_paragraph_order_with_concurrent_results() -> None:
    chunk = EvidenceChunk(
        source_id="src-news",
        source_type=SourceType.NEWS,
        text="Important news item.",
    )
    paragraphs = [
        AnalysisParagraph(
            section_name="Recent Developments",
            text="First draft.",
            chunk_ids=[chunk.chunk_id],
            verification_status=VerificationStatus.FAIL,
        ),
        AnalysisParagraph(
            section_name="Recent Developments",
            text="Second draft.",
            chunk_ids=[chunk.chunk_id],
            verification_status=VerificationStatus.FAIL,
        ),
    ]
    for paragraph in paragraphs:
        paragraph.checker_verdicts.append(
            CheckerVerdict(
                paragraph_id=paragraph.paragraph_id,
                status=VerificationStatus.FAIL,
                issues=[_issue(paragraph.paragraph_id)],
            )
        )

    section = ReportSection(
        section_name="Recent Developments",
        section_order=1,
        paragraphs=list(paragraphs),
        verification_status=VerificationStatus.FAIL,
    )
    state = {
        "analysis_sections": [section],
        "evidence_chunks": [chunk],
        "calculations": [],
        "revision_count": 0,
    }

    def _fake_revise(*, paragraph, verdict, chunks, calc_results, iteration):
        del verdict, chunks, calc_results
        if paragraph.text == "First draft.":
            time.sleep(0.05)
        revised = paragraph.model_copy(deep=True)
        revised.text = f"{paragraph.text} revised"
        record = RevisionRecord(
            paragraph_id=paragraph.paragraph_id,
            iteration=iteration,
            original_text=paragraph.text,
            revised_text=revised.text,
            changes_summary="Concurrent test revision.",
        )
        return revised, record

    with patch.object(
        verification_agent_module.ReviserAgent,
        "revise",
        side_effect=_fake_revise,
    ):
        updated = revise_sections(state)

    updated_paragraphs = updated["analysis_sections"][0].paragraphs
    assert [para.text for para in updated_paragraphs] == [
        "First draft. revised",
        "Second draft. revised",
    ]
    assert updated["revision_count"] == 1
