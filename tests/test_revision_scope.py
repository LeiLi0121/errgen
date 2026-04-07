from __future__ import annotations

import importlib
import time
from unittest.mock import patch

from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
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
from errgen.nodes.analysis_agent import (
    select_calcs_for_section_name,
    select_chunks_for_section_name,
)
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


def test_select_calcs_filters_business_and_risk_sections() -> None:
    revenue_yoy = CalculationResult(
        calc_id="calc-yoy",
        operation="yoy_growth_table",
        inputs={},
        result={"table": []},
        formula_description="YoY growth table",
        description="Revenue YoY Growth Table",
    )
    current_ratio = CalculationResult(
        calc_id="calc-current",
        operation="current_ratio",
        inputs={},
        result=1.5,
        formula_description="current assets / current liabilities",
        description="Liquidity ratio",
    )
    price_return = CalculationResult(
        calc_id="calc-price",
        operation="growth_rate",
        inputs={},
        result=0.12,
        formula_description="price return",
        description="ABC 1y price return",
    )
    excess_return = CalculationResult(
        calc_id="calc-diff",
        operation="difference",
        inputs={},
        result=0.03,
        formula_description="excess return",
        description="ABC 1y excess return vs SPY",
    )

    all_calcs = [revenue_yoy, current_ratio, price_return, excess_return]

    business = select_calcs_for_section_name(
        "Business & Competitive Analysis",
        all_calcs,
    )
    risk = select_calcs_for_section_name("Risk Analysis", all_calcs)

    assert [calc.calc_id for calc in business] == ["calc-price", "calc-diff"]
    assert [calc.calc_id for calc in risk] == [
        "calc-current",
        "calc-price",
        "calc-diff",
    ]


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


def test_revise_sections_passes_section_scoped_calcs_to_reviser() -> None:
    paragraph = AnalysisParagraph(
        section_name="Risk Analysis",
        text="Draft risk paragraph.",
        verification_status=VerificationStatus.FAIL,
    )
    verdict = CheckerVerdict(
        paragraph_id=paragraph.paragraph_id,
        status=VerificationStatus.FAIL,
        issues=[_issue(paragraph.paragraph_id)],
    )
    paragraph.checker_verdicts.append(verdict)
    section = ReportSection(
        section_name="Risk Analysis",
        section_order=1,
        paragraphs=[paragraph],
        verification_status=VerificationStatus.FAIL,
    )
    state = {
        "analysis_sections": [section],
        "evidence_chunks": [],
        "calculations": [
            CalculationResult(
                calc_id="calc-yoy",
                operation="yoy_growth_table",
                inputs={},
                result={"table": []},
                formula_description="YoY growth table",
                description="Revenue YoY Growth Table",
            ),
            CalculationResult(
                calc_id="calc-current",
                operation="current_ratio",
                inputs={},
                result=1.5,
                formula_description="current assets / current liabilities",
                description="Liquidity ratio",
            ),
            CalculationResult(
                calc_id="calc-price",
                operation="growth_rate",
                inputs={},
                result=0.12,
                formula_description="price return",
                description="ABC 1y price return",
            ),
        ],
        "revision_count": 0,
    }
    captured_calc_ids: list[str] = []

    def _fake_revise(*, paragraph, verdict, chunks, calc_results, iteration):
        del paragraph, verdict, chunks, iteration
        captured_calc_ids.extend(calc.calc_id for calc in calc_results)
        revision_record = RevisionRecord(
            paragraph_id="risk-para",
            iteration=1,
            original_text="Draft risk paragraph.",
            revised_text="Draft risk paragraph.",
            changes_summary="No-op test revision.",
        )
        return state["analysis_sections"][0].paragraphs[0], revision_record

    with patch.object(
        verification_agent_module.ReviserAgent,
        "revise",
        side_effect=_fake_revise,
    ):
        revise_sections(state)

    assert captured_calc_ids == ["calc-current", "calc-price"]


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


def test_verification_agent_uses_round_snapshot_for_concurrent_checks() -> None:
    overview_paragraph = AnalysisParagraph(
        section_name="Company Overview",
        text="Verified context paragraph.",
        chunk_ids=[],
        verification_status=VerificationStatus.PASS,
    )
    first_fail = AnalysisParagraph(
        section_name="Recent Developments",
        text="First fail.",
        chunk_ids=[],
        verification_status=VerificationStatus.FAIL,
    )
    second_fail = AnalysisParagraph(
        section_name="Recent Developments",
        text="Second fail.",
        chunk_ids=[],
        verification_status=VerificationStatus.FAIL,
    )
    sections = [
        ReportSection(
            section_name="Company Overview",
            section_order=1,
            paragraphs=[overview_paragraph],
            verification_status=VerificationStatus.PASS,
        ),
        ReportSection(
            section_name="Recent Developments",
            section_order=2,
            paragraphs=[first_fail, second_fail],
            verification_status=VerificationStatus.FAIL,
        ),
    ]
    state = {
        "analysis_sections": sections,
        "evidence_chunks": [],
        "calculations": [],
        "revision_count": 0,
    }
    seen_contexts: list[str] = []

    def _fake_check(*, paragraph, chunks, calc_results, iteration, as_of_date, verified_context):
        del paragraph, chunks, calc_results, iteration, as_of_date
        seen_contexts.append(verified_context)
        return CheckerVerdict(
            paragraph_id="dummy",
            status=VerificationStatus.PASS,
            issues=[],
        )

    with patch.object(
        verification_agent_module.CheckerAgent,
        "check",
        side_effect=_fake_check,
    ):
        updated = verification_agent_module.verification_agent(state)

    assert len(seen_contexts) == 2
    assert seen_contexts[0] == seen_contexts[1]
    assert "Verified context paragraph." in seen_contexts[0]
    updated_paragraphs = updated["analysis_sections"][1].paragraphs
    assert all(para.verification_status == VerificationStatus.PASS for para in updated_paragraphs)
