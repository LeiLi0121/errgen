"""
Evaluate already-generated baseline/full prediction reruns against GT.

This script does NOT rerun prediction generation. It reuses existing rerun
directories and computes:
1. Exact rating match against GT
2. Optional LLM-as-judge comparison of prediction text vs GT investment overview

Typical usage:

python3 scripts/evaluate_prediction_reruns.py \
  --queries evaluation/reports_extracted/errgen_queries.jsonl \
  --baseline-runs-root evaluation/runs/baseline_prediction_reruns \
  --full-runs-root evaluation/runs/full_prediction_reruns \
  --output-dir evaluation/results_prediction_reruns_eval \
  --judge-model qwen3.5-plus \
  --max-workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from errgen.config import Config
from evaluation.io import filter_queries, index_runs_by_query, load_queries
from scripts.rerun_prediction_vs_gt_batch import (
    _build_record,
    _normalize_rating,
    _summarize,
    _summarize_llm_judge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing baseline/full prediction reruns against GT.",
    )
    parser.add_argument(
        "--queries",
        default="evaluation/reports_extracted/errgen_queries.jsonl",
        help="Path to the evaluation query set.",
    )
    parser.add_argument("--baseline-runs-root", required=True)
    parser.add_argument("--full-runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--report-id",
        action="append",
        default=[],
        help="Optional report_id filter. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent evaluation workers.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional LLM judge model for prediction-vs-GT pairwise evaluation.",
    )
    parser.add_argument(
        "--gt-field",
        default="narrative_text_path",
        choices=["narrative_text_path", "full_text_path"],
        help="Ground-truth text field used to extract the GT prediction section.",
    )
    parser.add_argument(
        "--skip-missing-runs",
        action="store_true",
        help="Skip samples whose baseline/full rerun artifacts are missing instead of failing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue when one sample fails instead of aborting the full batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judge_model:
        Config.validate_required()

    queries = load_queries(args.queries)
    selected = filter_queries(
        queries,
        report_ids=set(args.report_id) if args.report_id else None,
        limit=args.limit,
    )

    baseline_index = index_runs_by_query(args.baseline_runs_root)
    full_index = index_runs_by_query(args.full_runs_root)

    paired_samples = []
    skipped_missing: list[str] = []
    for sample in selected:
        baseline_run_dir = baseline_index.get(sample.query)
        full_run_dir = full_index.get(sample.query)
        if baseline_run_dir and full_run_dir:
            paired_samples.append((sample, str(baseline_run_dir), str(full_run_dir)))
            continue
        message = (
            f"Missing rerun dirs for sample {sample.report_id}. "
            f"baseline={baseline_run_dir}, full={full_run_dir}"
        )
        if not args.skip_missing_runs:
            raise FileNotFoundError(message)
        skipped_missing.append(sample.report_id)
        print(f"[skip missing] {sample.report_id} ({sample.ticker})")

    if not paired_samples:
        raise FileNotFoundError("No paired prediction reruns found for the selected samples.")

    records: list[dict] = []
    future_map = {}
    max_workers = max(1, min(args.max_workers, len(paired_samples)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (sample, baseline_run_dir, full_run_dir) in enumerate(paired_samples, start=1):
            print(f"[{idx}/{len(paired_samples)}] queued {sample.report_id} ({sample.ticker})")
            payload = {
                "gt_rating": _normalize_rating(sample.rating),
                "baseline_run_dir": baseline_run_dir,
                "full_run_dir": full_run_dir,
            }
            future = executor.submit(
                _build_record,
                sample,
                payload,
                args.gt_field,
                args.judge_model,
            )
            future_map[future] = sample.report_id

        completed = 0
        for future in as_completed(future_map):
            sample_id = future_map[future]
            try:
                record = future.result()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[error] {sample_id} -> {exc}")
                continue
            if record is None:
                continue
            records.append(record)
            completed += 1
            print(f"[done {completed}/{len(paired_samples)}] evaluated: {sample_id}")

    records.sort(key=lambda item: item["report_id"])

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_sample.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "n_selected_samples": len(selected),
        "n_paired_samples": len(paired_samples),
        "n_completed_comparisons": len(records),
        "ground_truth_field": args.gt_field,
        "skipped_missing_runs": skipped_missing,
        "baseline": _summarize(records, "baseline"),
        "full": _summarize(records, "full"),
    }
    if args.judge_model:
        summary["llm_judge_model"] = args.judge_model
        summary["baseline_llm_judge"] = _summarize_llm_judge(records, "baseline")
        summary["full_llm_judge"] = _summarize_llm_judge(records, "full")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved evaluation results to {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
