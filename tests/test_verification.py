"""
Tests for the checker and reviser agents using mocked LLM calls.

These tests verify the structural logic of CheckerAgent and ReviserAgent
without making real API calls.  The LLM responses are mocked via unittest.mock.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    EvidenceChunk,
    IssueSeverity,
    IssueType,
    SourceType,
    VerificationStatus,
)
from errgen.verification.checker import CheckerAgent
from errgen.verification.reviser import ReviserAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk() -> EvidenceChunk:
    return EvidenceChunk(
        source_id="src-001",
        source_type=SourceType.INCOME_STATEMENT,
        text="NVDA FY2024 Revenue: $60,922,000,000. YoY growth: 122%.",
        period="FY 2024",
        field_name="revenue",
        numeric_value=60_922_000_000,
        unit="USD",
    )


@pytest.fixture
def sample_calc() -> CalculationResult:
    return CalculationResult(
        calc_id="calc-001",
        operation="growth_rate",
        inputs={"current": 60922, "previous": 26974},
        result=1.2597,
        formula_description="(60922-26974)/26974 = 1.2597",
        description="NVDA FY2024 revenue YoY growth",
    )


@pytest.fixture
def passing_paragraph(sample_chunk: EvidenceChunk) -> AnalysisParagraph:
    return AnalysisParagraph(
        section_name="Financial Analysis",
        text="NVIDIA's revenue grew approximately 122% year-over-year in FY2024, "
             "reaching $60.9 billion.",
        chunk_ids=[sample_chunk.chunk_id],
        calc_ids=["calc-001"],
    )


@pytest.fixture
def failing_paragraph(sample_chunk: EvidenceChunk) -> AnalysisParagraph:
    return AnalysisParagraph(
        section_name="Financial Analysis",
        text="NVIDIA's revenue grew 500% year-over-year in FY2024, "
             "making it the most profitable company ever.",  # both claims wrong
        chunk_ids=[sample_chunk.chunk_id],
        calc_ids=[],
    )


# ---------------------------------------------------------------------------
# CheckerAgent tests
# ---------------------------------------------------------------------------


def _mock_llm_pass_response() -> dict:
    return {"status": "pass", "issues": []}


def _mock_llm_fail_response(para_id: str) -> dict:
    return {
        "status": "fail",
        "issues": [
            {
                "issue_type": "numerical_error",
                "severity": "critical",
                "offending_span": "grew 500%",
                "explanation": "The cited chunk shows 122% growth, not 500%.",
                "relevant_chunk_ids": [],
                "relevant_calc_ids": [],
                "recommended_fix": "Change '500%' to '122%'.",
            },
            {
                "issue_type": "hallucination",
                "severity": "major",
                "offending_span": "most profitable company ever",
                "explanation": "No evidence supports this superlative claim.",
                "relevant_chunk_ids": [],
                "relevant_calc_ids": [],
                "recommended_fix": "Remove or qualify the superlative claim.",
            },
        ],
    }


class TestCheckerAgent:
    def test_pass_verdict(self, passing_paragraph, sample_chunk, sample_calc):
        checker = CheckerAgent()
        with patch("errgen.verification.checker.chat_json", return_value=_mock_llm_pass_response()):
            verdict = checker.check(
                paragraph=passing_paragraph,
                chunks=[sample_chunk],
                calc_results=[sample_calc],
            )
        assert verdict.status == VerificationStatus.PASS
        assert len(verdict.issues) == 0
        assert verdict.paragraph_id == passing_paragraph.paragraph_id

    def test_fail_verdict_with_issues(self, failing_paragraph, sample_chunk, sample_calc):
        checker = CheckerAgent()
        mock_resp = _mock_llm_fail_response(failing_paragraph.paragraph_id)
        with patch("errgen.verification.checker.chat_json", return_value=mock_resp):
            verdict = checker.check(
                paragraph=failing_paragraph,
                chunks=[sample_chunk],
                calc_results=[sample_calc],
            )
        assert verdict.status == VerificationStatus.FAIL
        assert len(verdict.issues) == 2
        issue_types = {i.issue_type for i in verdict.issues}
        assert IssueType.NUMERICAL_ERROR in issue_types
        assert IssueType.HALLUCINATION in issue_types

    def test_critical_issue_forces_fail_even_if_llm_says_pass(
        self, failing_paragraph, sample_chunk
    ):
        """If LLM says 'pass' but there are critical issues, we override to fail."""
        checker = CheckerAgent()
        inconsistent_response = {
            "status": "pass",  # LLM inconsistently says pass
            "issues": [
                {
                    "issue_type": "numerical_error",
                    "severity": "critical",
                    "offending_span": "500%",
                    "explanation": "Wrong number.",
                    "relevant_chunk_ids": [],
                    "relevant_calc_ids": [],
                    "recommended_fix": "Fix the number.",
                }
            ],
        }
        with patch("errgen.verification.checker.chat_json", return_value=inconsistent_response):
            verdict = checker.check(
                paragraph=failing_paragraph,
                chunks=[sample_chunk],
                calc_results=[],
            )
        assert verdict.status == VerificationStatus.FAIL

    def test_checker_handles_llm_failure(self, passing_paragraph, sample_chunk):
        checker = CheckerAgent()
        with patch(
            "errgen.verification.checker.chat_json",
            side_effect=RuntimeError("OpenAI down"),
        ):
            verdict = checker.check(
                paragraph=passing_paragraph,
                chunks=[sample_chunk],
                calc_results=[],
            )
        assert verdict.status == VerificationStatus.UNRESOLVED
        assert len(verdict.issues) == 1
        assert "failed" in verdict.issues[0].explanation.lower()

    def test_unknown_issue_type_defaults_to_unsupported_claim(
        self, failing_paragraph, sample_chunk
    ):
        checker = CheckerAgent()
        response_with_bad_type = {
            "status": "fail",
            "issues": [
                {
                    "issue_type": "completely_made_up_type",
                    "severity": "major",
                    "explanation": "Test",
                    "relevant_chunk_ids": [],
                    "relevant_calc_ids": [],
                    "recommended_fix": "Fix it.",
                }
            ],
        }
        with patch("errgen.verification.checker.chat_json", return_value=response_with_bad_type):
            verdict = checker.check(
                paragraph=failing_paragraph,
                chunks=[sample_chunk],
                calc_results=[],
            )
        assert verdict.issues[0].issue_type == IssueType.UNSUPPORTED_CLAIM


# ---------------------------------------------------------------------------
# ReviserAgent tests
# ---------------------------------------------------------------------------


class TestReviserAgent:
    def _make_verdict(self, para_id: str, issues: list[dict]):
        from errgen.models import CheckerIssue, CheckerVerdict

        checker_issues = [
            CheckerIssue(
                issue_type=IssueType(i["issue_type"]),
                severity=IssueSeverity(i["severity"]),
                paragraph_id=para_id,
                offending_span=i.get("offending_span"),
                explanation=i["explanation"],
                recommended_fix=i["recommended_fix"],
            )
            for i in issues
        ]
        return CheckerVerdict(
            paragraph_id=para_id,
            status=VerificationStatus.FAIL,
            issues=checker_issues,
        )

    def test_reviser_produces_revised_paragraph(
        self, failing_paragraph, sample_chunk, sample_calc
    ):
        reviser = ReviserAgent()
        verdict = self._make_verdict(
            failing_paragraph.paragraph_id,
            [
                {
                    "issue_type": "numerical_error",
                    "severity": "critical",
                    "offending_span": "500%",
                    "explanation": "Should be 122%.",
                    "recommended_fix": "Use 122%.",
                }
            ],
        )

        mock_revision = {
            "text": "NVIDIA's revenue grew approximately 122% year-over-year in FY2024.",
            "chunk_ids": [sample_chunk.chunk_id],
            "calc_ids": ["calc-001"],
            "changes_summary": "Corrected growth rate from 500% to 122%.",
        }
        with patch("errgen.verification.reviser.chat_json", return_value=mock_revision):
            revised_para, record = reviser.revise(
                paragraph=failing_paragraph,
                verdict=verdict,
                chunks=[sample_chunk],
                calc_results=[sample_calc],
                iteration=1,
            )

        assert "122%" in revised_para.text
        assert sample_chunk.chunk_id in revised_para.chunk_ids
        assert record.iteration == 1
        assert "500%" not in revised_para.text

    def test_reviser_no_blocking_issues_returns_unchanged(
        self, passing_paragraph, sample_chunk
    ):
        reviser = ReviserAgent()
        from errgen.models import CheckerVerdict

        verdict_with_only_minor = CheckerVerdict(
            paragraph_id=passing_paragraph.paragraph_id,
            status=VerificationStatus.FAIL,
            issues=[],  # no blocking issues
        )
        revised_para, record = reviser.revise(
            paragraph=passing_paragraph,
            verdict=verdict_with_only_minor,
            chunks=[sample_chunk],
            calc_results=[],
            iteration=1,
        )
        assert revised_para.text == passing_paragraph.text
        assert record.changes_summary == "No blocking issues – no changes made."

    def test_reviser_handles_llm_failure(self, failing_paragraph, sample_chunk):
        reviser = ReviserAgent()
        verdict = self._make_verdict(
            failing_paragraph.paragraph_id,
            [
                {
                    "issue_type": "hallucination",
                    "severity": "major",
                    "explanation": "Invented fact.",
                    "recommended_fix": "Remove it.",
                }
            ],
        )
        with patch(
            "errgen.verification.reviser.chat_json",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            revised_para, record = reviser.revise(
                paragraph=failing_paragraph,
                verdict=verdict,
                chunks=[sample_chunk],
                calc_results=[],
                iteration=1,
            )
        # Should return original text unchanged
        assert revised_para.text == failing_paragraph.text
        assert "failed" in record.changes_summary.lower()

    def test_reviser_strips_invalid_chunk_ids(self, failing_paragraph, sample_chunk):
        """Chunk IDs invented by the LLM that don't exist should be dropped."""
        reviser = ReviserAgent()
        verdict = self._make_verdict(
            failing_paragraph.paragraph_id,
            [
                {
                    "issue_type": "numerical_error",
                    "severity": "critical",
                    "explanation": "Wrong number.",
                    "recommended_fix": "Fix.",
                }
            ],
        )
        mock_revision = {
            "text": "Corrected paragraph text.",
            "chunk_ids": [sample_chunk.chunk_id, "nonexistent-chunk-id"],
            "calc_ids": ["nonexistent-calc-id"],
            "changes_summary": "Fixed.",
        }
        with patch("errgen.verification.reviser.chat_json", return_value=mock_revision):
            revised_para, record = reviser.revise(
                paragraph=failing_paragraph,
                verdict=verdict,
                chunks=[sample_chunk],
                calc_results=[],
                iteration=1,
            )
        # Nonexistent IDs should be stripped
        assert "nonexistent-chunk-id" not in revised_para.chunk_ids
        assert "nonexistent-calc-id" not in revised_para.calc_ids
        # Valid chunk ID should be kept
        assert sample_chunk.chunk_id in revised_para.chunk_ids

    def test_reviser_recovers_ids_from_text_and_removes_inline_aliases(
        self, failing_paragraph, sample_chunk, sample_calc
    ):
        reviser = ReviserAgent()
        verdict = self._make_verdict(
            failing_paragraph.paragraph_id,
            [
                {
                    "issue_type": "numerical_error",
                    "severity": "critical",
                    "explanation": "Wrong number.",
                    "recommended_fix": "Fix.",
                }
            ],
        )
        mock_revision = {
            "text": "Corrected paragraph text (C001, K001).",
            "chunk_ids": [],
            "calc_ids": [],
            "changes_summary": "Fixed.",
        }
        with patch("errgen.verification.reviser.chat_json", return_value=mock_revision):
            revised_para, _ = reviser.revise(
                paragraph=failing_paragraph,
                verdict=verdict,
                chunks=[sample_chunk],
                calc_results=[sample_calc],
                iteration=1,
            )

        assert revised_para.text == "Corrected paragraph text."
        assert revised_para.chunk_ids == [sample_chunk.chunk_id]
        assert revised_para.calc_ids == [sample_calc.calc_id]
