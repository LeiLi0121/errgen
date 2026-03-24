from evaluation.schemas import (
    IssueFinding,
    IssueFindings,
    PairwiseEvaluation,
    PairwiseJudgeOutput,
    QualityScores,
    ScoredDimension,
)
from evaluation.scoring import (
    compute_adjusted_win_rate,
    compute_final_label,
    compute_q_score,
    compute_tier,
    count_issue_findings,
    passes_accuracy_gate,
    resolve_pairwise_outcome,
    summarize_pairwise_results,
)


def _dim(score: int) -> ScoredDimension:
    return ScoredDimension(score=score, reason="ok")


def test_count_issue_findings_and_gate():
    findings = IssueFindings(
        unsupported_claims=[
            IssueFinding(severity="severe", explanation="x"),
            IssueFinding(severity="minor", explanation="y"),
        ],
        numerical_errors=[IssueFinding(severity="minor", explanation="z")],
    )

    counts, severe = count_issue_findings(findings)

    assert counts["unsupported_claims"] == 2
    assert counts["numerical_errors"] == 1
    assert severe["unsupported_claims"] == 1
    assert not passes_accuracy_gate(severe)


def test_q_score_and_tier():
    quality = QualityScores(
        financial_numeric=_dim(5),
        news=_dim(4),
        cmi=_dim(4),
        invest=_dim(4),
        risk=_dim(4),
        writing=_dim(3),
    )

    q_score = compute_q_score(quality)
    assert q_score == 4.2
    assert compute_tier(True, q_score) == "Tier A"
    assert compute_final_label(True, q_score) == "High Quality"
    assert compute_tier(False, q_score) == "Fail"
    assert compute_final_label(False, q_score) == "Fail"


def test_pairwise_summary():
    pairwise_results = [
        PairwiseEvaluation(
            sample_id="A",
            ticker="AAA US",
            as_of_date="2025-01-01",
            model_a="baseline",
            model_b="errgen",
            judge_model="judge",
            order_ab_winner="errgen",
            order_ba_winner="errgen",
            final_outcome="errgen",
            order_ab=PairwiseJudgeOutput(
                factual_comparison="x",
                quality_comparison="y",
                winner="second_report",
                rationale="z",
            ),
            order_ba=PairwiseJudgeOutput(
                factual_comparison="x",
                quality_comparison="y",
                winner="first_report",
                rationale="z",
            ),
        ),
        PairwiseEvaluation(
            sample_id="B",
            ticker="BBB US",
            as_of_date="2025-01-01",
            model_a="baseline",
            model_b="errgen",
            judge_model="judge",
            order_ab_winner="Tie",
            order_ba_winner="Tie",
            final_outcome="Tie",
            order_ab=PairwiseJudgeOutput(
                factual_comparison="x",
                quality_comparison="y",
                winner="tie",
                rationale="z",
            ),
            order_ba=PairwiseJudgeOutput(
                factual_comparison="x",
                quality_comparison="y",
                winner="tie",
                rationale="z",
            ),
        ),
    ]

    assert resolve_pairwise_outcome("errgen", "errgen") == "errgen"
    assert resolve_pairwise_outcome("errgen", "baseline") == "Tie"
    assert compute_adjusted_win_rate(1, 0, 1) == 0.75

    summary = summarize_pairwise_results(pairwise_results, preferred_model="errgen")
    assert summary == {
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "adjusted_win_rate": 0.75,
    }
