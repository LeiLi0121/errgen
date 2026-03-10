"""
Evaluation module – extension point for future research evaluation.

Currently provides:
  - metric interfaces (abstract base classes)
  - stub implementations for key evaluation dimensions
  - a report evaluator that aggregates metrics

Actual implementations (model-based, human, or automatic) can be plugged in
by subclassing the interfaces below without touching the pipeline.
"""

from evaluation.metrics import (
    CitationPrecision,
    CitationRecall,
    FactualGroundingScore,
    NumericalCorrectnessScore,
    ReportEvaluator,
)

__all__ = [
    "CitationPrecision",
    "CitationRecall",
    "FactualGroundingScore",
    "NumericalCorrectnessScore",
    "ReportEvaluator",
]
