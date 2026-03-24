"""
Tests for data model serialisation, validation, and helper methods.

All tests are offline and do not require API keys.
"""

import json
from datetime import datetime

import pytest

from errgen.models import (
    AnalysisParagraph,
    CalculationRequest,
    CalculationResult,
    CheckerIssue,
    CheckerVerdict,
    Citation,
    EvidenceChunk,
    ExtractedFact,
    FinalReport,
    IssueType,
    IssueSeverity,
    ReportTable,
    ReportSection,
    RunArtifact,
    SourceMetadata,
    SourceType,
    UserRequest,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# UserRequest
# ---------------------------------------------------------------------------


def test_user_request_defaults():
    req = UserRequest(raw_text="Test report for NVDA", ticker="NVDA")
    assert req.market == "US"
    assert req.language == "en"
    assert req.request_id  # auto-generated UUID
    assert isinstance(req.created_at, datetime)


def test_user_request_serialisation():
    req = UserRequest(
        raw_text="NVIDIA report",
        ticker="NVDA",
        as_of_date="2025-01",
        focus_areas=["AI chips", "financial analysis"],
    )
    data = json.loads(req.model_dump_json())
    assert data["ticker"] == "NVDA"
    assert data["as_of_date"] == "2025-01"
    assert "AI chips" in data["focus_areas"]


# ---------------------------------------------------------------------------
# EvidenceChunk
# ---------------------------------------------------------------------------


def test_evidence_chunk_creation():
    src = SourceMetadata(
        source_type=SourceType.INCOME_STATEMENT,
        api_source="fmp",
        ticker="NVDA",
    )
    chunk = EvidenceChunk(
        source_id=src.source_id,
        source_type=SourceType.INCOME_STATEMENT,
        text="NVIDIA FY2024 Revenue: $60,922,000,000",
        period="FY 2024",
        field_name="revenue",
        numeric_value=60_922_000_000,
        unit="USD",
    )
    assert chunk.chunk_id
    assert chunk.numeric_value == 60_922_000_000


def test_evidence_chunk_roundtrip():
    chunk = EvidenceChunk(
        source_id="src-123",
        source_type=SourceType.NEWS,
        text="NVIDIA announces H100 chip.",
        metadata={"url": "https://example.com", "date": "2024-03-01"},
    )
    data = chunk.model_dump(mode="json")
    restored = EvidenceChunk(**data)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == chunk.text
    assert restored.metadata["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# AnalysisParagraph
# ---------------------------------------------------------------------------


def test_analysis_paragraph_default_status():
    para = AnalysisParagraph(
        section_name="Financial Analysis",
        text="Revenue grew 122% YoY.",
        chunk_ids=["chunk-001"],
    )
    assert para.verification_status == VerificationStatus.FAIL
    assert para.paragraph_id


def test_analysis_paragraph_with_citations():
    citation = Citation(
        chunk_id="chunk-001",
        source_id="src-001",
        source_type=SourceType.INCOME_STATEMENT,
        text_snippet="Revenue: $60.9B",
    )
    para = AnalysisParagraph(
        section_name="Financial Analysis",
        text="Revenue grew 122% YoY to $60.9B.",
        citations=[citation],
        chunk_ids=["chunk-001"],
        calc_ids=["calc-001"],
        verification_status=VerificationStatus.PASS,
    )
    assert len(para.citations) == 1
    assert para.citations[0].chunk_id == "chunk-001"


# ---------------------------------------------------------------------------
# CheckerIssue and CheckerVerdict
# ---------------------------------------------------------------------------


def test_checker_issue_creation():
    issue = CheckerIssue(
        issue_type=IssueType.NUMERICAL_ERROR,
        severity=IssueSeverity.CRITICAL,
        paragraph_id="para-001",
        offending_span="grew by 150%",
        explanation="The cited chunk shows 122% growth, not 150%.",
        relevant_chunk_ids=["chunk-001"],
        recommended_fix="Change '150%' to '122%' and cite the correct chunk.",
    )
    assert issue.issue_id
    assert issue.severity == IssueSeverity.CRITICAL


def test_checker_verdict_pass():
    verdict = CheckerVerdict(
        paragraph_id="para-001",
        status=VerificationStatus.PASS,
        issues=[],
        iteration=0,
    )
    assert verdict.status == VerificationStatus.PASS
    assert len(verdict.issues) == 0


def test_checker_verdict_fail_with_issues():
    issue = CheckerIssue(
        issue_type=IssueType.HALLUCINATION,
        severity=IssueSeverity.MAJOR,
        paragraph_id="para-001",
        explanation="Claim about market share has no supporting chunk.",
        recommended_fix="Remove or cite the market share claim.",
    )
    verdict = CheckerVerdict(
        paragraph_id="para-001",
        status=VerificationStatus.FAIL,
        issues=[issue],
        iteration=1,
    )
    assert len(verdict.issues) == 1
    assert verdict.iteration == 1


# ---------------------------------------------------------------------------
# RunArtifact helpers
# ---------------------------------------------------------------------------


def test_run_artifact_chunk_by_id():
    chunk = EvidenceChunk(
        source_id="src-1",
        source_type=SourceType.INCOME_STATEMENT,
        text="Revenue data",
    )
    req = UserRequest(raw_text="Test", ticker="NVDA")
    artifact = RunArtifact(request=req, evidence_chunks=[chunk])

    found = artifact.chunk_by_id(chunk.chunk_id)
    assert found is not None
    assert found.chunk_id == chunk.chunk_id

    not_found = artifact.chunk_by_id("nonexistent-id")
    assert not_found is None


def test_run_artifact_calc_by_id():
    calc = CalculationResult(
        calc_id="calc-001",
        operation="growth_rate",
        inputs={"current": 200, "previous": 100},
        result=1.0,
        formula_description="(200-100)/100 = 1.0",
        description="Revenue growth",
    )
    req = UserRequest(raw_text="Test", ticker="NVDA")
    artifact = RunArtifact(request=req, calculation_results=[calc])

    found = artifact.calc_by_id("calc-001")
    assert found is not None
    assert found.result == 1.0

    not_found = artifact.calc_by_id("nonexistent")
    assert not_found is None


# ---------------------------------------------------------------------------
# FinalReport serialisation
# ---------------------------------------------------------------------------


def test_final_report_empty():
    req = UserRequest(raw_text="Test", ticker="NVDA")
    report = FinalReport(request=req)
    assert report.overall_status == VerificationStatus.FAIL
    assert len(report.sections) == 0
    assert len(report.tables) == 0

    # Ensure full JSON serialisation works
    data = json.loads(report.model_dump_json())
    assert data["request"]["ticker"] == "NVDA"


def test_final_report_with_table_roundtrip():
    req = UserRequest(raw_text="Test", ticker="NVDA")
    table = ReportTable(
        title="Company Snapshot",
        columns=["Metric", "Value"],
        rows=[["Ticker", "NVDA"], ["Exchange", "NASDAQ"]],
    )
    report = FinalReport(request=req, tables=[table])
    payload = report.model_dump(mode="json")
    restored = FinalReport(**payload)
    assert restored.tables[0].title == "Company Snapshot"
    assert restored.tables[0].rows[1][1] == "NASDAQ"


# ---------------------------------------------------------------------------
# CalculationRequest / Result
# ---------------------------------------------------------------------------


def test_calculation_request_auto_id():
    req1 = CalculationRequest(
        operation="growth_rate",
        inputs={"current": 100, "previous": 80},
        description="Test",
    )
    req2 = CalculationRequest(
        operation="growth_rate",
        inputs={"current": 100, "previous": 80},
        description="Test",
    )
    assert req1.calc_id != req2.calc_id


def test_calculation_result_error_field():
    result = CalculationResult(
        calc_id="c-001",
        operation="growth_rate",
        inputs={"current": 100, "previous": 0},
        result=float("nan"),
        formula_description="Error: division by zero",
        description="Failed calc",
        error="Previous value is zero",
    )
    assert result.error is not None
    assert "zero" in result.error
