"""
Aggregate single-report evaluation results into model-level score summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FACTUAL_DIMS = (
    "claim_support",
    "numerical_accuracy",
    "citation_precision",
    "temporal_validity",
    "consistency",
)

QUALITY_DIMS = (
    "financial_numeric",
    "news",
    "cmi",
    "invest",
    "risk",
    "writing",
)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _load_single_results(single_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(single_dir.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def _summarize_model(items: list[dict[str, Any]]) -> dict[str, Any]:
    pass_items = [item for item in items if item["accuracy_gate_passed"]]

    factual_all = {
        dim: _mean([item["accuracy"][dim]["score"] for item in items])
        for dim in FACTUAL_DIMS
    }
    factual_pass_only = {
        dim: _mean([item["accuracy"][dim]["score"] for item in pass_items])
        for dim in FACTUAL_DIMS
    }

    factual_overall_all = _mean(
        [
            sum(item["accuracy"][dim]["score"] for dim in FACTUAL_DIMS) / len(FACTUAL_DIMS)
            for item in items
        ]
    )
    factual_overall_pass_only = _mean(
        [
            sum(item["accuracy"][dim]["score"] for dim in FACTUAL_DIMS) / len(FACTUAL_DIMS)
            for item in pass_items
        ]
    )

    quality_all = {
        dim: _mean([item["quality"][dim]["score"] for item in items])
        for dim in QUALITY_DIMS
    }
    quality_pass_only = {
        dim: _mean([item["quality"][dim]["score"] for item in pass_items])
        for dim in QUALITY_DIMS
    }

    return {
        "n_samples": len(items),
        "accuracy_gate_passed": sum(1 for item in items if item["accuracy_gate_passed"]),
        "q_score_avg_all": _mean([item["q_score"] for item in items]),
        "q_score_avg_pass_only": _mean([item["q_score"] for item in pass_items]),
        "factual_avg_all": factual_all,
        "factual_avg_pass_only": factual_pass_only,
        "factual_overall_avg_all": factual_overall_all,
        "factual_overall_avg_pass_only": factual_overall_pass_only,
        "quality_avg_all": quality_all,
        "quality_avg_pass_only": quality_pass_only,
    }


def aggregate_single_results(single_dir: Path) -> dict[str, Any]:
    results = _load_single_results(single_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item["model_name"]].append(item)

    return {
        "single_dir": str(single_dir),
        "n_single_results": len(results),
        "models": {
            model_name: _summarize_model(items)
            for model_name, items in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate factual and quality averages from single evaluation results."
    )
    parser.add_argument(
        "--single-dir",
        default="evaluation/results_gated_split_v2_rerun/single",
        help="Directory containing single-report evaluation JSON files.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    args = parser.parse_args()

    summary = aggregate_single_results(Path(args.single_dir))
    if args.pretty:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
