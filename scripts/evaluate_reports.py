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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from evaluation.io import (
    filter_queries,
    index_runs_by_query,
    load_queries,
    load_run_bundle,
    load_single_evaluation,
)
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent evaluation workers.",
    )
    parser.add_argument(
        "--skip-missing-runs",
        action="store_true",
        help="Skip samples whose baseline/full run artifacts are missing instead of failing.",
    )
    parser.add_argument(
        "--pairwise-only",
        action="store_true",
        help="Reuse existing single-report evaluation JSON files and run pairwise only.",
    )
    parser.add_argument(
        "--single-dir",
        default=None,
        help="Directory containing existing single-report evaluation JSON files. "
             "Defaults to <output-dir>/single.",
    )
    return parser.parse_args()


def _evaluate_sample(
    sample,
    baseline_run_dir,
    full_run_dir,
    judge_model: str | None,
    baseline_label: str,
    full_label: str,
):
    judge = LLMReportJudge(judge_model=judge_model)
    baseline_bundle = load_run_bundle(baseline_run_dir)
    full_bundle = load_run_bundle(full_run_dir)

    with ThreadPoolExecutor(max_workers=2) as executor:
        factual_future_map = {
            executor.submit(
                judge.judge_factual,
                bundle=baseline_bundle,
                sample=sample,
                model_name=baseline_label,
            ): "baseline",
            executor.submit(
                judge.judge_factual,
                bundle=full_bundle,
                sample=sample,
                model_name=full_label,
            ): "full",
        }
        factual_states = {}
        for future in as_completed(factual_future_map):
            factual_states[factual_future_map[future]] = future.result()

    quality_results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        quality_future_map = {}
        if factual_states["baseline"].accuracy_gate_passed:
            quality_future_map[
                executor.submit(
                    judge.judge_quality,
                    bundle=baseline_bundle,
                    sample=sample,
                    model_name=baseline_label,
                )
            ] = "baseline"
        if factual_states["full"].accuracy_gate_passed:
            quality_future_map[
                executor.submit(
                    judge.judge_quality,
                    bundle=full_bundle,
                    sample=sample,
                    model_name=full_label,
                )
            ] = "full"
        for future in as_completed(quality_future_map):
            quality_results[quality_future_map[future]] = future.result()

    baseline_eval = judge.build_single_evaluation(
        bundle=baseline_bundle,
        sample=sample,
        model_name=baseline_label,
        factual_state=factual_states["baseline"],
        quality_result=quality_results.get("baseline"),
    )
    full_eval = judge.build_single_evaluation(
        bundle=full_bundle,
        sample=sample,
        model_name=full_label,
        factual_state=factual_states["full"],
        quality_result=quality_results.get("full"),
    )
    pairwise_eval = judge.judge_pairwise(
        sample=sample,
        model_a=baseline_label,
        model_b=full_label,
        eval_a=baseline_eval,
        eval_b=full_eval,
        bundle_a=baseline_bundle,
        bundle_b=full_bundle,
    )

    return sample, baseline_eval, full_eval, pairwise_eval


def _evaluate_pairwise_only_sample(
    sample,
    baseline_run_dir,
    full_run_dir,
    single_dir: Path,
    judge_model: str | None,
    baseline_label: str,
    full_label: str,
):
    judge = LLMReportJudge(judge_model=judge_model)
    baseline_bundle = load_run_bundle(baseline_run_dir)
    full_bundle = load_run_bundle(full_run_dir)

    baseline_eval_path = single_dir / f"{sample.report_id}_{baseline_label}.json"
    full_eval_path = single_dir / f"{sample.report_id}_{full_label}.json"
    if not baseline_eval_path.exists() or not full_eval_path.exists():
        raise FileNotFoundError(
            f"Missing single evaluation results for sample {sample.report_id}. "
            f"baseline_eval={baseline_eval_path if baseline_eval_path.exists() else None}, "
            f"full_eval={full_eval_path if full_eval_path.exists() else None}"
        )

    baseline_eval = load_single_evaluation(baseline_eval_path)
    full_eval = load_single_evaluation(full_eval_path)
    pairwise_eval = judge.judge_pairwise(
        sample=sample,
        model_a=baseline_label,
        model_b=full_label,
        eval_a=baseline_eval,
        eval_b=full_eval,
        bundle_a=baseline_bundle,
        bundle_b=full_bundle,
    )
    return sample, baseline_eval, full_eval, pairwise_eval


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

    output_dir = Path(args.output_dir)
    single_dir = output_dir / "single"
    pairwise_dir = output_dir / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    if not args.pairwise_only:
        single_dir.mkdir(parents=True, exist_ok=True)
    reuse_single_dir = Path(args.single_dir) if args.single_dir else single_dir

    pairwise_results = []
    skipped_missing: list[str] = []
    skipped_missing_single: list[str] = []
    runnable = []
    for sample in selected:
        baseline_run_dir = baseline_index.get(sample.query)
        full_run_dir = full_index.get(sample.query)
        if baseline_run_dir and full_run_dir:
            runnable.append((sample, baseline_run_dir, full_run_dir))
            continue
        message = (
            f"Missing runs for sample {sample.report_id}. "
            f"baseline={baseline_run_dir}, full={full_run_dir}"
        )
        if not args.skip_missing_runs:
            raise FileNotFoundError(message)
        skipped_missing.append(sample.report_id)
        print(f"[skip missing] {sample.report_id} ({sample.ticker})")

    total = len(runnable)
    if total == 0:
        raise FileNotFoundError("No runnable samples found after filtering missing runs.")

    max_workers = max(1, min(args.max_workers, total))
    future_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (sample, baseline_run_dir, full_run_dir) in enumerate(runnable, start=1):
            print(f"[{idx}/{total}] queued {sample.report_id} ({sample.ticker})")
            if args.pairwise_only:
                future = executor.submit(
                    _evaluate_pairwise_only_sample,
                    sample,
                    baseline_run_dir,
                    full_run_dir,
                    reuse_single_dir,
                    args.judge_model,
                    args.baseline_label,
                    args.full_label,
                )
            else:
                future = executor.submit(
                    _evaluate_sample,
                    sample,
                    baseline_run_dir,
                    full_run_dir,
                    args.judge_model,
                    args.baseline_label,
                    args.full_label,
                )
            future_map[future] = sample.report_id

        completed = 0
        for future in as_completed(future_map):
            sample_id = future_map[future]
            try:
                sample, baseline_eval, full_eval, pairwise_eval = future.result()
            except FileNotFoundError as exc:
                if not args.pairwise_only:
                    raise
                skipped_missing_single.append(sample_id)
                print(f"[skip missing single] {sample_id}: {exc}")
                continue
            completed += 1
            pairwise_results.append(pairwise_eval)

            if not args.pairwise_only:
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
            print(f"[done {completed}/{total}] saved: {sample_id}")

    summary = summarize_pairwise_results(pairwise_results, preferred_model=args.full_label)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "full_label": args.full_label,
                "n_samples": len(pairwise_results),
                "skipped_missing_runs": skipped_missing,
                "skipped_missing_single": skipped_missing_single,
                "pairwise_only": args.pairwise_only,
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
