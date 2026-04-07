from pathlib import Path

from evaluation.judges import LLMReportJudge
from evaluation.schemas import EvalQuery, RunBundle
from errgen.models import (
    AnalysisParagraph,
    FinalReport,
    ReportSection,
    UserRequest,
    VerificationStatus,
)


def _sample_bundle() -> RunBundle:
    request = UserRequest(
        raw_text="Write an equity research report for TEST.",
        ticker="TEST",
        company_name="Test Corp",
        as_of_date="2025-01-01",
    )
    paragraph = AnalysisParagraph(
        section_name="Company Overview",
        text="Test Corp operates a software platform.",
        chunk_ids=["C001"],
        calc_ids=["K001"],
        verification_status=VerificationStatus.PASS,
    )
    report = FinalReport(
        request=request,
        sections=[
            ReportSection(
                section_name="Company Overview",
                section_order=1,
                paragraphs=[paragraph],
                verification_status=VerificationStatus.PASS,
            )
        ],
    )
    return RunBundle(
        run_dir=Path("/tmp/fake-run"),
        run_id="fake-run",
        request=request,
        report=report,
        report_text="# Test report",
        evidence_chunks=[],
        calculation_results=[],
        manifest={},
    )


def _sample_query() -> EvalQuery:
    return EvalQuery(
        report_id="sample-1",
        company_name="Test Corp",
        ticker="TEST US",
        as_of_date="2025-01-01",
        query="Write an equity research report for Test Corp.",
    )


def test_judge_single_runs_quality_only_after_passing_gate(monkeypatch):
    responses = [
        {
            "accuracy": {
                "claim_support": {"score": 5, "reason": "ok"},
                "numerical_accuracy": {"score": 5, "reason": "ok"},
                "citation_precision": {"score": 5, "reason": "ok"},
                "temporal_validity": {"score": 5, "reason": "ok"},
                "consistency": {"score": 5, "reason": "ok"},
            },
            "issue_findings": {
                "unsupported_claims": [],
                "numerical_errors": [],
                "citation_mismatches": [],
                "temporal_violations": [],
                "inconsistencies": [],
            },
        },
        {
            "quality": {
                "financial_numeric": {"score": 4, "reason": "ok"},
                "news": {"score": 4, "reason": "ok"},
                "cmi": {"score": 4, "reason": "ok"},
                "invest": {"score": 4, "reason": "ok"},
                "risk": {"score": 4, "reason": "ok"},
                "writing": {"score": 4, "reason": "ok"},
            },
            "summary": {
                "primary_strength": "Strong structure.",
                "primary_weakness": "Needs more depth.",
            },
        },
    ]
    calls = []

    def fake_chat_json(messages, model=None, temperature=0.0):
        del model, temperature
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr("evaluation.judges.chat_json", fake_chat_json)

    judge = LLMReportJudge(judge_model="judge-model")
    result = judge.judge_single(_sample_bundle(), _sample_query(), "errgen")

    assert len(calls) == 2
    assert result.accuracy_gate_passed is True
    assert result.q_score == 4.0
    assert result.quality.writing.score == 4
    assert result.summary.primary_strength == "Strong structure."


def test_judge_single_skips_quality_when_gate_fails(monkeypatch):
    calls = []

    def fake_chat_json(messages, model=None, temperature=0.0):
        del messages, model, temperature
        calls.append("called")
        return {
            "accuracy": {
                "claim_support": {"score": 1, "reason": "bad"},
                "numerical_accuracy": {"score": 5, "reason": "ok"},
                "citation_precision": {"score": 2, "reason": "weak"},
                "temporal_validity": {"score": 1, "reason": "future data"},
                "consistency": {"score": 4, "reason": "ok"},
            },
            "issue_findings": {
                "unsupported_claims": [],
                "numerical_errors": [],
                "citation_mismatches": [],
                "temporal_violations": [
                    {
                        "severity": "severe",
                        "explanation": "Used post as-of evidence.",
                        "section": "Company Overview",
                        "offending_span": "future event",
                    }
                ],
                "inconsistencies": [],
            },
        }

    monkeypatch.setattr("evaluation.judges.chat_json", fake_chat_json)

    judge = LLMReportJudge(judge_model="judge-model")
    result = judge.judge_single(_sample_bundle(), _sample_query(), "errgen")

    assert len(calls) == 1
    assert result.accuracy_gate_passed is False
    assert result.q_score == 0.0
    assert result.quality.financial_numeric.score == 1
    assert "Accuracy Gate" in result.quality.financial_numeric.reason
    assert "Accuracy Gate" in result.summary.primary_weakness


def test_judge_pairwise_compares_sections_and_aggregates(monkeypatch):
    base_bundle = _sample_bundle()
    full_bundle = _sample_bundle()

    extra_base = AnalysisParagraph(
        section_name="Risk Analysis",
        text="Base risk section.",
        chunk_ids=[],
        calc_ids=[],
        verification_status=VerificationStatus.PASS,
    )
    extra_full = AnalysisParagraph(
        section_name="Risk Analysis",
        text="Full risk section.",
        chunk_ids=[],
        calc_ids=[],
        verification_status=VerificationStatus.PASS,
    )
    base_bundle.report.sections.append(
        ReportSection(
            section_name="Risk Analysis",
            section_order=2,
            paragraphs=[extra_base],
            verification_status=VerificationStatus.PASS,
        )
    )
    full_bundle.report.sections.append(
        ReportSection(
            section_name="Risk Analysis",
            section_order=2,
            paragraphs=[extra_full],
            verification_status=VerificationStatus.PASS,
        )
    )

    factual_pass = {
        "accuracy": {
            "claim_support": {"score": 5, "reason": "ok"},
            "numerical_accuracy": {"score": 5, "reason": "ok"},
            "citation_precision": {"score": 5, "reason": "ok"},
            "temporal_validity": {"score": 5, "reason": "ok"},
            "consistency": {"score": 5, "reason": "ok"},
        },
        "issue_findings": {
            "unsupported_claims": [],
            "numerical_errors": [],
            "citation_mismatches": [],
            "temporal_violations": [],
            "inconsistencies": [],
        },
    }
    quality_pass = {
        "quality": {
            "financial_numeric": {"score": 4, "reason": "ok"},
            "news": {"score": 4, "reason": "ok"},
            "cmi": {"score": 4, "reason": "ok"},
            "invest": {"score": 4, "reason": "ok"},
            "risk": {"score": 4, "reason": "ok"},
            "writing": {"score": 4, "reason": "ok"},
        },
        "summary": {
            "primary_strength": "Strong structure.",
            "primary_weakness": "Needs more depth.",
        },
    }

    def fake_chat_json(messages, model=None, temperature=0.0):
        del model, temperature
        user_prompt = messages[-1]["content"]
        if "Section: Company Overview" in user_prompt:
            winner = "second_report" if "=== FIRST REPORT: errgen-baseline ===" in user_prompt else "first_report"
        elif "Section: Risk Analysis" in user_prompt:
            winner = "tie"
        elif "=== REPORT WITH EVIDENCE PACK ===" in user_prompt:
            return factual_pass
        elif "=== REPORT ===" in user_prompt:
            return quality_pass
        else:
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        return {
            "factual_comparison": "section comparison",
            "quality_comparison": "section quality",
            "winner": winner,
            "rationale": "section rationale",
        }

    monkeypatch.setattr("evaluation.judges.chat_json", fake_chat_json)

    judge = LLMReportJudge(judge_model="judge-model")
    baseline_eval = judge.judge_single(base_bundle, _sample_query(), "errgen-baseline")
    full_eval = judge.judge_single(full_bundle, _sample_query(), "errgen")
    result = judge.judge_pairwise(
        sample=_sample_query(),
        model_a="errgen-baseline",
        model_b="errgen",
        eval_a=baseline_eval,
        eval_b=full_eval,
        bundle_a=base_bundle,
        bundle_b=full_bundle,
    )

    assert len(result.section_results) == 2
    assert [item.section_name for item in result.section_results] == [
        "Company Overview",
        "Risk Analysis",
    ]
    assert result.section_results[0].final_outcome == "errgen"
    assert result.section_results[1].final_outcome == "Tie"
    assert result.section_outcome_counts == {
        "errgen-baseline": 0,
        "errgen": 1,
        "Tie": 1,
    }
    assert result.final_outcome == "errgen"
