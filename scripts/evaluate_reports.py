"""
Evaluate baseline vs full ERRGen runs using an LLM-as-judge workflow.

Typical usage:

python scripts/evaluate_reports.py \
  --queries evaluation/reports_extracted/errgen_queries.jsonl \
  --baseline-runs-root runs/baseline \
  --full-runs-root runs/full \
  --output-dir evaluation/results \
  --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from evaluation.io import filter_queries, index_runs_by_query, load_queries, load_run_bundle
from evaluation.judges import LLMReportJudge
from evaluation.scoring import summarize_pairwise_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs full ERRGen runs with an LLM judge.",
    )
    parser.add_argument(
        "--queries",
        default="evaluation/reports_extracted/errgen_queries.jsonl",
        help="Path to the evaluation query set.",
    )
    parser.add_argument("--baseline-runs-root", required=True)
    parser.add_argument("--full-runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--report-id",
        action="append",
        default=[],
        help="Optional report_id filter. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--baseline-label",
        default="errgen-baseline",
        help="Label used in saved evaluation outputs.",
    )
    parser.add_argument(
        "--full-label",
        default="errgen",
        help="Label used in saved evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    selected = filter_queries(
        queries,
        report_ids=set(args.report_id) if args.report_id else None,
        limit=args.limit,
    )

    baseline_index = index_runs_by_query(args.baseline_runs_root)
    full_index = index_runs_by_query(args.full_runs_root)
    judge = LLMReportJudge(judge_model=args.judge_model)

    output_dir = Path(args.output_dir)
    single_dir = output_dir / "single"
    pairwise_dir = output_dir / "pairwise"
    single_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    pairwise_results = []

    for sample in selected:
        baseline_run_dir = baseline_index.get(sample.query)
        full_run_dir = full_index.get(sample.query)
        if not baseline_run_dir or not full_run_dir:
            raise FileNotFoundError(
                f"Missing runs for sample {sample.report_id}. "
                f"baseline={baseline_run_dir}, full={full_run_dir}"
            )

        baseline_bundle = load_run_bundle(baseline_run_dir)
        full_bundle = load_run_bundle(full_run_dir)

        baseline_eval = judge.judge_single(
            bundle=baseline_bundle,
            sample=sample,
            model_name=args.baseline_label,
        )
        full_eval = judge.judge_single(
            bundle=full_bundle,
            sample=sample,
            model_name=args.full_label,
        )
        pairwise_eval = judge.judge_pairwise(
            sample=sample,
            model_a=args.baseline_label,
            model_b=args.full_label,
            eval_a=baseline_eval,
            eval_b=full_eval,
            bundle_a=baseline_bundle,
            bundle_b=full_bundle,
        )
        pairwise_results.append(pairwise_eval)

        (single_dir / f"{sample.report_id}_{args.baseline_label}.json").write_text(
            json.dumps(baseline_eval.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (single_dir / f"{sample.report_id}_{args.full_label}.json").write_text(
            json.dumps(full_eval.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (pairwise_dir / f"{sample.report_id}.json").write_text(
            json.dumps(pairwise_eval.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = summarize_pairwise_results(pairwise_results, preferred_model=args.full_label)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "full_label": args.full_label,
                "n_samples": len(pairwise_results),
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved evaluation results to {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
