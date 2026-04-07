"""
Pure scoring helpers for ERRGen evaluation.
"""

from __future__ import annotations

from evaluation.schemas import IssueFindings, PairwiseEvaluation, QualityScores

QUALITY_WEIGHTS = {
    "financial_numeric": 0.30,
    "news": 0.15,
    "cmi": 0.15,
    "invest": 0.15,
    "risk": 0.15,
    "writing": 0.10,
}


def count_issue_findings(issue_findings: IssueFindings) -> tuple[dict[str, int], dict[str, int]]:
    counts = {
        "unsupported_claims": len(issue_findings.unsupported_claims),
        "numerical_errors": len(issue_findings.numerical_errors),
        "citation_mismatches": len(issue_findings.citation_mismatches),
        "temporal_violations": len(issue_findings.temporal_violations),
        "inconsistencies": len(issue_findings.inconsistencies),
    }
    severe_counts = {
        "unsupported_claims": sum(
            1 for finding in issue_findings.unsupported_claims
            if finding.severity == "severe"
        ),
        "numerical_errors": sum(
            1 for finding in issue_findings.numerical_errors
            if finding.severity == "severe"
        ),
        "citation_mismatches": sum(
            1 for finding in issue_findings.citation_mismatches
            if finding.severity == "severe"
        ),
        "temporal_violations": sum(
            1 for finding in issue_findings.temporal_violations
            if finding.severity == "severe"
        ),
        "inconsistencies": sum(
            1 for finding in issue_findings.inconsistencies
            if finding.severity == "severe"
        ),
    }
    return counts, severe_counts


def passes_accuracy_gate(severe_counts: dict[str, int]) -> bool:
    return (
        severe_counts["numerical_errors"] == 0
        and severe_counts["unsupported_claims"] == 0
        and severe_counts["temporal_violations"] == 0
        and severe_counts["inconsistencies"] == 0
    )


def compute_q_score(quality: QualityScores) -> float:
    raw = (
        QUALITY_WEIGHTS["financial_numeric"] * quality.financial_numeric.score
        + QUALITY_WEIGHTS["news"] * quality.news.score
        + QUALITY_WEIGHTS["cmi"] * quality.cmi.score
        + QUALITY_WEIGHTS["invest"] * quality.invest.score
        + QUALITY_WEIGHTS["risk"] * quality.risk.score
        + QUALITY_WEIGHTS["writing"] * quality.writing.score
    )
    return round(raw, 4)


def compute_tier(accuracy_gate_passed: bool, q_score: float) -> str:
    if not accuracy_gate_passed:
        return "Fail"
    if q_score >= 4.2:
        return "Tier A"
    if q_score >= 3.5:
        return "Tier B"
    return "Tier C"


def compute_final_label(accuracy_gate_passed: bool, q_score: float) -> str:
    if not accuracy_gate_passed:
        return "Fail"
    if q_score >= 4.2:
        return "High Quality"
    return "Pass"


def resolve_pairwise_outcome(order_ab_winner: str, order_ba_winner: str) -> str:
    if order_ab_winner == order_ba_winner:
        return order_ab_winner
    return "Tie"


def aggregate_pairwise_winners(
    outcomes: list[str],
    first_label: str,
    second_label: str,
) -> str:
    first_wins = sum(1 for outcome in outcomes if outcome == first_label)
    second_wins = sum(1 for outcome in outcomes if outcome == second_label)
    if first_wins > second_wins:
        return first_label
    if second_wins > first_wins:
        return second_label
    return "Tie"


def compute_adjusted_win_rate(wins: int, losses: int, ties: int) -> float:
    total = wins + losses + ties
    if total == 0:
        return 0.0
    return round((wins + 0.5 * ties) / total, 4)


def summarize_pairwise_results(
    pairwise_results: list[PairwiseEvaluation],
    preferred_model: str,
) -> dict[str, float | int]:
    wins = sum(1 for item in pairwise_results if item.final_outcome == preferred_model)
    losses = sum(
        1 for item in pairwise_results
        if item.final_outcome not in {preferred_model, "Tie"}
    )
    ties = sum(1 for item in pairwise_results if item.final_outcome == "Tie")
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "adjusted_win_rate": compute_adjusted_win_rate(wins, losses, ties),
    }
