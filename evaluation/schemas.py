"""
Structured schemas for ERRGen evaluation outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from errgen.models import CalculationResult, EvidenceChunk, FinalReport, UserRequest


class EvalQuery(BaseModel):
    report_id: str
    pdf_path: Optional[str] = None
    full_text_path: Optional[str] = None
    narrative_text_path: Optional[str] = None
    company_name: str
    ticker: str
    as_of_date: str
    query: str
    llm_model: Optional[str] = None
    company: Optional[str] = None
    report_date: Optional[str] = None
    headline: Optional[str] = None
    rating: Optional[str] = None
    target_price: Optional[float] = None
    target_currency: Optional[str] = None


class ScoredDimension(BaseModel):
    score: int
    reason: str

    @field_validator("score")
    @classmethod
    def _validate_score(cls, value: int) -> int:
        if value < 1 or value > 5:
            raise ValueError("score must be between 1 and 5")
        return value


class IssueFinding(BaseModel):
    severity: Literal["severe", "minor"]
    explanation: str
    section: Optional[str] = None
    offending_span: Optional[str] = None


class IssueFindings(BaseModel):
    unsupported_claims: list[IssueFinding] = Field(default_factory=list)
    numerical_errors: list[IssueFinding] = Field(default_factory=list)
    citation_mismatches: list[IssueFinding] = Field(default_factory=list)
    temporal_violations: list[IssueFinding] = Field(default_factory=list)
    inconsistencies: list[IssueFinding] = Field(default_factory=list)


class AccuracyScores(BaseModel):
    claim_support: ScoredDimension
    numerical_accuracy: ScoredDimension
    citation_precision: ScoredDimension
    temporal_validity: ScoredDimension
    consistency: ScoredDimension


class QualityScores(BaseModel):
    financial_numeric: ScoredDimension
    news: ScoredDimension
    cmi: ScoredDimension
    invest: ScoredDimension
    risk: ScoredDimension
    writing: ScoredDimension


class EvaluationSummary(BaseModel):
    primary_strength: str
    primary_weakness: str


class SingleReportJudgeOutput(BaseModel):
    accuracy: AccuracyScores
    issue_findings: IssueFindings
    quality: QualityScores
    summary: EvaluationSummary


class FactualJudgeOutput(BaseModel):
    accuracy: AccuracyScores
    issue_findings: IssueFindings


class QualityJudgeOutput(BaseModel):
    quality: QualityScores
    summary: EvaluationSummary


class SingleReportEvaluation(BaseModel):
    sample_id: str
    model_name: str
    ticker: str
    as_of_date: str
    run_id: Optional[str] = None
    judge_model: str
    accuracy_gate_passed: bool
    accuracy: AccuracyScores
    issue_findings: IssueFindings = Field(default_factory=IssueFindings)
    error_counts: dict[str, int]
    severe_error_counts: dict[str, int]
    quality: QualityScores
    q_score: float
    tier: Literal["Tier A", "Tier B", "Tier C", "Fail"]
    final_label: Literal["Fail", "Pass", "High Quality"]
    summary: EvaluationSummary


class PairwiseJudgeOutput(BaseModel):
    factual_comparison: str
    quality_comparison: str
    winner: Literal["first_report", "second_report", "tie"]
    rationale: str


class SectionPairwiseEvaluation(BaseModel):
    section_name: str
    order_ab_winner: str
    order_ba_winner: str
    final_outcome: str
    order_ab: PairwiseJudgeOutput
    order_ba: PairwiseJudgeOutput


class PairwiseEvaluation(BaseModel):
    sample_id: str
    ticker: str
    as_of_date: str
    model_a: str
    model_b: str
    judge_model: str
    order_ab_winner: str
    order_ba_winner: str
    final_outcome: str
    order_ab: PairwiseJudgeOutput
    order_ba: PairwiseJudgeOutput
    section_results: list[SectionPairwiseEvaluation] = Field(default_factory=list)
    section_outcome_counts: dict[str, int] = Field(default_factory=dict)


@dataclass
class RunBundle:
    run_dir: Path
    run_id: str
    request: UserRequest
    report: FinalReport
    report_text: str
    evidence_chunks: list[EvidenceChunk]
    calculation_results: list[CalculationResult]
    manifest: dict[str, Any]
