"""
Batch runner for ERRGen evaluation experiments.

This script can:
1. Run the baseline and/or full errgen pipeline over a selected query subset.
2. Save run artifacts under evaluation/runs by default.
3. Optionally launch the LLM-as-judge evaluation over the same subset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import sys
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from errgen.config import Config
from errgen.graph import run
from evaluation.io import filter_queries, index_runs_by_query, load_queries, load_run_bundle
from evaluation.judges import LLMReportJudge
from evaluation.scoring import summarize_pairwise_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline/full ERRGen batches and optionally evaluate them.",
    )
    parser.add_argument(
        "--queries",
        default="evaluation/reports_extracted/errgen_queries.jsonl",
        help="Path to the evaluation query set.",
    )
    parser.add_argument(
        "--runs-root",
        default="evaluation/runs",
        help="Root directory for generated run artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        default="evaluation/results",
        help="Directory for saved evaluation outputs.",
    )
    parser.add_argument(
        "--modes",
        choices=["baseline", "full", "both"],
        default="both",
        help="Which generation modes to run.",
    )
    parser.add_argument(
        "--report-id",
        action="append",
        default=[],
        help="Optional report_id filter. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N filtered samples before running.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation when a run for the same query already exists.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run single-report and pairwise evaluation after generation.",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--baseline-label", default="errgen-baseline")
    parser.add_argument("--full-label", default="errgen")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent evaluation workers.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO/DEBUG-style pipeline logging during batch runs.",
    )
    return parser.parse_args()


def _selected_modes(mode_arg: str) -> list[str]:
    if mode_arg == "both":
        return ["baseline", "full"]
    return [mode_arg]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_project_root / path).resolve()


def _prepare_queries(args: argparse.Namespace):
    queries = load_queries(_resolve_path(args.queries))
    selected = filter_queries(
        queries,
        report_ids=set(args.report_id) if args.report_id else None,
        limit=None,
    )
    if args.offset:
        selected = selected[args.offset:]
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def _generate_runs(args: argparse.Namespace, selected_queries) -> dict[str, dict[str, str]]:
    runs_root = _resolve_path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    selected_modes = _selected_modes(args.modes)
    existing_indices = {
        mode: index_runs_by_query(runs_root / mode)
        for mode in selected_modes
    }
    generated: dict[str, dict[str, str]] = {}

    for idx, sample in enumerate(selected_queries, start=1):
        generated[sample.report_id] = {}
        print(f"[{idx}/{len(selected_queries)}] {sample.report_id} ({sample.ticker})")
        for mode in selected_modes:
            existing_run_dir = existing_indices[mode].get(sample.query)
            if args.skip_existing and existing_run_dir:
                generated[sample.report_id][mode] = str(existing_run_dir)
                print(f"  - {mode}: skip existing -> {existing_run_dir.name}")
                continue

            mode_runs_dir = runs_root / mode
            mode_runs_dir.mkdir(parents=True, exist_ok=True)
            Config.RUNS_DIR = str(mode_runs_dir)

            final_state = run(sample.query, mode=mode)
            run_id = final_state.get("run_id")
            if not run_id:
                raise RuntimeError(f"No run_id returned for {sample.report_id} ({mode})")

            run_dir = mode_runs_dir / run_id
            generated[sample.report_id][mode] = str(run_dir)
            existing_indices[mode][sample.query] = run_dir
            print(f"  - {mode}: completed -> {run_id}")

    return generated


def _evaluate_runs(
    args: argparse.Namespace,
    selected_queries,
) -> Optional[dict]:
    runs_root = _resolve_path(args.runs_root)
    baseline_root = runs_root / "baseline"
    full_root = runs_root / "full"
    if not baseline_root.exists() or not full_root.exists():
        raise FileNotFoundError(
            "Both baseline and full run directories must exist for evaluation."
        )

    baseline_index = index_runs_by_query(baseline_root)
    full_index = index_runs_by_query(full_root)
    results_dir = _resolve_path(args.results_dir)
    single_dir = results_dir / "single"
    pairwise_dir = results_dir / "pairwise"
    single_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    pairwise_results = []
    total = len(selected_queries)
    max_workers = max(1, min(args.max_workers, total))
    future_map = {}

    def _evaluate_sample(sample):
        judge = LLMReportJudge(judge_model=args.judge_model)
        baseline_run_dir = baseline_index.get(sample.query)
        full_run_dir = full_index.get(sample.query)
        if not baseline_run_dir or not full_run_dir:
            raise FileNotFoundError(
                f"Missing paired runs for sample {sample.report_id}: "
                f"baseline={baseline_run_dir}, full={full_run_dir}"
            )

        baseline_bundle = load_run_bundle(baseline_run_dir)
        full_bundle = load_run_bundle(full_run_dir)

        with ThreadPoolExecutor(max_workers=2) as executor:
            factual_future_map = {
                executor.submit(
                    judge.judge_factual,
                    bundle=baseline_bundle,
                    sample=sample,
                    model_name=args.baseline_label,
                ): "baseline",
                executor.submit(
                    judge.judge_factual,
                    bundle=full_bundle,
                    sample=sample,
                    model_name=args.full_label,
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
                        model_name=args.baseline_label,
                    )
                ] = "baseline"
            if factual_states["full"].accuracy_gate_passed:
                quality_future_map[
                    executor.submit(
                        judge.judge_quality,
                        bundle=full_bundle,
                        sample=sample,
                        model_name=args.full_label,
                    )
                ] = "full"
            for future in as_completed(quality_future_map):
                quality_results[quality_future_map[future]] = future.result()

        baseline_eval = judge.build_single_evaluation(
            bundle=baseline_bundle,
            sample=sample,
            model_name=args.baseline_label,
            factual_state=factual_states["baseline"],
            quality_result=quality_results.get("baseline"),
        )
        full_eval = judge.build_single_evaluation(
            bundle=full_bundle,
            sample=sample,
            model_name=args.full_label,
            factual_state=factual_states["full"],
            quality_result=quality_results.get("full"),
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
        return sample, baseline_eval, full_eval, pairwise_eval

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, sample in enumerate(selected_queries, start=1):
            print(f"[eval queued {idx}/{total}] {sample.report_id}")
            future = executor.submit(_evaluate_sample, sample)
            future_map[future] = sample.report_id

        completed = 0
        for future in as_completed(future_map):
            sample_id = future_map[future]
            sample, baseline_eval, full_eval, pairwise_eval = future.result()
            completed += 1
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
            print(f"[eval done {completed}/{total}] {sample_id}")

    summary = summarize_pairwise_results(pairwise_results, preferred_model=args.full_label)
    payload = {
        "baseline_label": args.baseline_label,
        "full_label": args.full_label,
        "n_samples": len(pairwise_results),
        **summary,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)
    selected_queries = _prepare_queries(args)
    if not selected_queries:
        print("No queries selected.")
        return

    _generate_runs(args, selected_queries)

    if args.evaluate:
        summary = _evaluate_runs(args, selected_queries)
        print("Evaluation summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
