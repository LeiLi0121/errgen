"""
Evaluation metric interfaces and stub implementations.

This module defines the evaluation framework for future research.
Current implementations are STUBS that return placeholder values.
They establish the interface contract so future work can plug in:
  - LLM-based evaluators (e.g., GPT-4 as judge)
  - Human annotation pipelines
  - Automatic reference-based metrics (e.g., BERTScore, FactScore)
  - Financial domain-specific scorers

Evaluation dimensions
---------------------
1. factual_grounding    – Are paragraph claims supported by evidence chunks?
2. citation_precision   – Do cited chunks actually support the claims?
3. citation_recall      – Are all key facts in evidence actually cited?
4. numerical_correctness – Are all numbers correct and consistent with calcs?
5. report_completeness  – Does the report cover all required sections?
6. consistency          – Do sections contradict each other?
7. style_quality        – Is language appropriate for an equity research report?
8. prediction_usefulness – Is the final recommendation grounded and actionable?
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from errgen.models import FinalReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base metric interface
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    metric_name: str
    score: float | None              # 0.0–1.0, or None if not computed
    details: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = True             # False once a real implementation exists
    notes: str = ""


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    name: str = "base_metric"

    @abstractmethod
    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        """
        Evaluate the report and return a MetricResult.

        Parameters
        ----------
        report:  The FinalReport to evaluate.
        **kwargs: Additional inputs (e.g., reference reports, human annotations).
        """
        ...


# ---------------------------------------------------------------------------
# Stub metric implementations
# ---------------------------------------------------------------------------


class FactualGroundingScore(BaseMetric):
    """
    Measures whether every factual claim in each paragraph is supported by
    at least one cited evidence chunk.

    Stub: returns the fraction of paragraphs that passed the checker.
    Real implementation: LLM-as-judge or human annotation per claim.
    """

    name = "factual_grounding"

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        from errgen.models import VerificationStatus

        total = 0
        passed = 0
        for section in report.sections:
            for para in section.paragraphs:
                total += 1
                if para.verification_status == VerificationStatus.PASS:
                    passed += 1

        score = passed / total if total > 0 else None
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={"total_paragraphs": total, "passed_paragraphs": passed},
            is_stub=True,
            notes=(
                "Stub: based on checker pass rate. "
                "Real implementation needs per-claim annotation."
            ),
        )


class CitationPrecision(BaseMetric):
    """
    Measures: of all chunk citations made, what fraction actually support
    the claim they are attached to.

    Stub: returns None (requires human annotation or LLM judge per citation).
    """

    name = "citation_precision"

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        total_citations = sum(
            len(para.citations)
            for section in report.sections
            for para in section.paragraphs
        )
        return MetricResult(
            metric_name=self.name,
            score=None,
            details={"total_citations": total_citations},
            is_stub=True,
            notes=(
                "Stub: requires LLM judge or human annotation to verify "
                "each citation's relevance to its paragraph claim."
            ),
        )


class CitationRecall(BaseMetric):
    """
    Measures: of all key facts present in the evidence pool, what fraction
    are actually cited in the report.

    Stub: returns None (requires reference annotation of key facts).
    """

    name = "citation_recall"

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=None,
            details={},
            is_stub=True,
            notes=(
                "Stub: requires a reference set of key facts from the evidence. "
                "Consider using NewsExtractor + FinancialExtractor output as proxy."
            ),
        )


class NumericalCorrectnessScore(BaseMetric):
    """
    Measures: are all numbers in the report consistent with cited evidence
    chunks and deterministic calculation results?

    Stub: returns the fraction of calculations that had no error.
    """

    name = "numerical_correctness"

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        calcs = report.calculation_appendix
        if not calcs:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"n_calculations": 0},
                is_stub=True,
                notes="No calculations in report.",
            )
        error_free = sum(1 for c in calcs if not c.error)
        score = error_free / len(calcs)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "n_calculations": len(calcs),
                "n_error_free": error_free,
                "errors": [c.error for c in calcs if c.error],
            },
            is_stub=True,
            notes="Stub: based on calculator error rate, not LLM number extraction.",
        )


class ReportCompletenessScore(BaseMetric):
    """
    Measures whether the report covers all expected sections.

    Stub: checks presence of expected section names.
    """

    name = "report_completeness"

    EXPECTED_SECTIONS = {
        "Company Overview",
        "Recent Developments",
        "Financial Analysis",
        "Business & Competitive Analysis",
        "Risk Analysis",
        "Investment Recommendation & Outlook",
    }

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        present = {s.section_name for s in report.sections}
        missing = self.EXPECTED_SECTIONS - present
        score = len(present & self.EXPECTED_SECTIONS) / len(self.EXPECTED_SECTIONS)
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={"present": sorted(present), "missing": sorted(missing)},
            is_stub=True,
        )


class ConsistencyScore(BaseMetric):
    """
    Measures: do sections contradict each other?

    Stub: returns None (requires cross-paragraph LLM consistency check).
    """

    name = "consistency"

    def evaluate(self, report: FinalReport, **kwargs: Any) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            score=None,
            details={},
            is_stub=True,
            notes="Stub: requires LLM or NLI-based cross-section consistency check.",
        )


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------


class ReportEvaluator:
    """
    Runs all registered metrics against a FinalReport and returns a summary.

    Usage:
        evaluator = ReportEvaluator()
        results = evaluator.evaluate(report)
        for r in results:
            print(r.metric_name, r.score)
    """

    def __init__(self, metrics: list[BaseMetric] | None = None) -> None:
        self.metrics: list[BaseMetric] = metrics or [
            FactualGroundingScore(),
            CitationPrecision(),
            CitationRecall(),
            NumericalCorrectnessScore(),
            ReportCompletenessScore(),
            ConsistencyScore(),
        ]

    def evaluate(self, report: FinalReport) -> list[MetricResult]:
        results: list[MetricResult] = []
        for metric in self.metrics:
            try:
                result = metric.evaluate(report)
                results.append(result)
                logger.info(
                    "Metric %s: score=%s (stub=%s)",
                    result.metric_name,
                    result.score,
                    result.is_stub,
                )
            except Exception as exc:
                logger.error("Metric %s failed: %s", metric.name, exc)
                results.append(
                    MetricResult(
                        metric_name=metric.name,
                        score=None,
                        is_stub=True,
                        notes=f"Evaluation error: {exc}",
                    )
                )
        return results
