"""
Rerun baseline and full prediction sections concurrently and compare ratings to GT.

This script:
1. Locates paired baseline/full runs for the selected query set.
2. Reruns only the prediction stage for both modes using saved upstream artifacts.
3. Saves fresh run artifacts to separate output roots.
4. Summarizes generated rating alignment against ground-truth ratings.

Typical usage:

python3 scripts/rerun_prediction_vs_gt_batch.py \
  --queries evaluation/reports_extracted/errgen_queries.jsonl \
  --baseline-runs-root evaluation/runs/baseline \
  --full-runs-root evaluation/runs/full \
  --baseline-output-root evaluation/runs/baseline_prediction_reruns \
  --full-output-root evaluation/runs/full_prediction_reruns \
  --output-dir evaluation/results_prediction_reruns \
  --max-workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from pydantic import BaseModel

from errgen.config import Config
from evaluation.io import filter_queries, index_runs_by_query, load_queries, load_run_bundle
from evaluation.schemas import EvalQuery
from errgen.llm import build_messages, chat_json
from scripts.rerun_prediction_from_run import rerun_prediction_for_run

GEN_RE = re.compile(r"\b(Buy|Hold|Sell)\b", re.I)
GT_PREDICTION_RE = re.compile(
    r"Investment Overview\s*:?\s*(.*)",
    re.I | re.S,
)


class PredictionPairwiseJudgeOutput(BaseModel):
    directional_alignment_comparison: str
    thesis_quality_comparison: str
    risk_balance_comparison: str
    actionability_comparison: str
    writing_comparison: str
    winner: str
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun baseline/full predictions concurrently and compare ratings to GT.",
    )
    parser.add_argument(
        "--queries",
        default="evaluation/reports_extracted/errgen_queries.jsonl",
        help="Path to the evaluation query set.",
    )
    parser.add_argument("--baseline-runs-root", required=True)
    parser.add_argument("--full-runs-root", required=True)
    parser.add_argument("--baseline-output-root", required=True)
    parser.add_argument("--full-output-root", required=True)
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
        help="Maximum number of concurrent rerun workers across both modes.",
    )
    parser.add_argument(
        "--suffix",
        default="pred-rerun",
        help="Suffix used when generating new run ids.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue when one sample fails instead of aborting the full batch.",
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
    return parser.parse_args()


def _normalize_rating(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().upper()
    if normalized == "FULLY VALUED":
        return "HOLD"
    return normalized


def _extract_generated_rating(run_dir: Path) -> str | None:
    bundle = load_run_bundle(run_dir)
    pred_section = next(
        (section for section in bundle.report.sections if section.section_name == "Investment Recommendation & Outlook"),
        None,
    )
    if not pred_section:
        return None
    text = "\n".join(paragraph.text.strip() for paragraph in pred_section.paragraphs)
    match = GEN_RE.search(text)
    return _normalize_rating(match.group(1) if match else None)


def _extract_generated_prediction_text(run_dir: Path) -> str | None:
    bundle = load_run_bundle(run_dir)
    pred_section = next(
        (
            section
            for section in bundle.report.sections
            if section.section_name == "Investment Recommendation & Outlook"
        ),
        None,
    )
    if not pred_section:
        return None
    text = "\n\n".join(paragraph.text.strip() for paragraph in pred_section.paragraphs).strip()
    return text or None


def _load_ground_truth_prediction_text(sample: EvalQuery, gt_field: str) -> str:
    text_path = getattr(sample, gt_field, None)
    if not text_path:
        raise FileNotFoundError(
            f"Sample {sample.report_id} missing ground-truth field {gt_field}"
        )
    path = Path(text_path)
    if not path.exists():
        subdir = "narrative_text" if gt_field == "narrative_text_path" else "full_text"
        fallback = _project_root / "evaluation" / "reports_extracted" / subdir / path.name
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth text not found for sample {sample.report_id}: {text_path}"
        )
    text = path.read_text(encoding="utf-8").strip()
    match = GT_PREDICTION_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _truncate(text: str, max_chars: int = 8000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def _prediction_pairwise_system_prompt() -> str:
    return dedent(
        """
        You are comparing a generated "Investment Recommendation & Outlook"
        section against a ground-truth investment overview for the same equity
        research task.

        Evaluate only these dimensions:
        - Directional alignment with the investment conclusion implied by the task
          and GT context
        - Thesis quality and analytical synthesis
        - Risk balance and acknowledgement of uncertainties
        - Investor actionability
        - Writing clarity and structure

        Important rules:
        - Ground truth is not automatically better. Judge the texts as written.
        - Do not reward a section merely for matching the GT rating if the logic is
          weak, generic, or poorly balanced.
        - A Hold recommendation can still be high quality if it is clearly justified.
        - Missing an explicit Buy/Hold/Sell label is a weakness, but if the stance is
          still clear from the text, assess it accordingly rather than treating it as
          an automatic failure.
        - Do not reward verbosity, boilerplate, or generic market language.
        - If one section is missing entirely and the other is present, prefer the
          present section unless its quality is unusably poor.
        - If the two sections are materially comparable overall, return tie.

        Return JSON only:
        {
          "directional_alignment_comparison": "...",
          "thesis_quality_comparison": "...",
          "risk_balance_comparison": "...",
          "actionability_comparison": "...",
          "writing_comparison": "...",
          "winner": "first_report",
          "rationale": "..."
        }
        """
    ).strip()


def _prediction_pairwise_user_prompt(
    sample: EvalQuery,
    first_label: str,
    second_label: str,
    first_report: str | None,
    second_report: str | None,
) -> str:
    headline = sample.headline or "N/A"
    rating = sample.rating or "N/A"
    target_price = (
        f"{sample.target_price} {sample.target_currency or ''}".strip()
        if sample.target_price is not None
        else "N/A"
    )
    first_text = first_report or "(missing prediction section)"
    second_text = second_report or "(missing prediction section)"
    return dedent(
        f"""
        Sample ID: {sample.report_id}
        Company: {sample.company_name}
        Ticker: {sample.ticker}
        As-of date: {sample.as_of_date}
        Query: {sample.query}

        Ground-truth metadata for context only:
        - Headline: {headline}
        - Rating: {rating}
        - Target price: {target_price}

        Compare the two prediction sections.

        === FIRST REPORT: {first_label} ===
        {_truncate(first_text)}

        === SECOND REPORT: {second_label} ===
        {_truncate(second_text)}
        """
    ).strip()


def _judge_prediction_once(
    sample: EvalQuery,
    first_label: str,
    second_label: str,
    first_report: str | None,
    second_report: str | None,
    judge_model: str,
) -> PredictionPairwiseJudgeOutput:
    messages = build_messages(
        _prediction_pairwise_system_prompt(),
        _prediction_pairwise_user_prompt(
            sample=sample,
            first_label=first_label,
            second_label=second_label,
            first_report=first_report,
            second_report=second_report,
        ),
    )
    raw = chat_json(messages, model=judge_model, temperature=0.0)
    return PredictionPairwiseJudgeOutput.model_validate(raw)


def _map_winner(winner: str, first_label: str, second_label: str) -> str:
    if winner == "first_report":
        return first_label
    if winner == "second_report":
        return second_label
    return "Tie"


def _resolve_outcome(order_ab: str, order_ba: str) -> str:
    if order_ab == order_ba:
        return order_ab
    return "Tie"


def _judge_prediction_vs_gt(
    sample: EvalQuery,
    generated_label: str,
    generated_text: str | None,
    gt_text: str,
    judge_model: str,
) -> dict:
    gt_label = "ground_truth"
    order_ab = _judge_prediction_once(
        sample=sample,
        first_label=generated_label,
        second_label=gt_label,
        first_report=generated_text,
        second_report=gt_text,
        judge_model=judge_model,
    )
    order_ba = _judge_prediction_once(
        sample=sample,
        first_label=gt_label,
        second_label=generated_label,
        first_report=gt_text,
        second_report=generated_text,
        judge_model=judge_model,
    )
    order_ab_winner = _map_winner(order_ab.winner, generated_label, gt_label)
    order_ba_winner = _map_winner(order_ba.winner, gt_label, generated_label)
    return {
        "judge_model": judge_model,
        "generated_label": generated_label,
        "ground_truth_label": gt_label,
        "order_ab_winner": order_ab_winner,
        "order_ba_winner": order_ba_winner,
        "final_outcome": _resolve_outcome(order_ab_winner, order_ba_winner),
        "order_ab": order_ab.model_dump(mode="json"),
        "order_ba": order_ba.model_dump(mode="json"),
    }


def _summarize(records: list[dict], model_key: str) -> dict:
    n = len(records)
    correct = sum(1 for item in records if item[f"{model_key}_match"])
    confusion = defaultdict(Counter)
    pred_dist = Counter()
    for item in records:
        confusion[item["gt_rating"]][item[f"{model_key}_rating"]] += 1
        pred_dist[item[f"{model_key}_rating"]] += 1
    return {
        "n_samples": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "pred_distribution": dict(pred_dist),
        "confusion": {gt: dict(counts) for gt, counts in confusion.items()},
    }


def _summarize_llm_judge(records: list[dict], model_key: str) -> dict:
    judged = [item for item in records if item.get(f"{model_key}_llm_judge")]
    n = len(judged)
    wins = sum(
        1
        for item in judged
        if item[f"{model_key}_llm_judge"]["final_outcome"] == model_key
    )
    losses = sum(
        1
        for item in judged
        if item[f"{model_key}_llm_judge"]["final_outcome"] == "ground_truth"
    )
    ties = sum(
        1
        for item in judged
        if item[f"{model_key}_llm_judge"]["final_outcome"] == "Tie"
    )
    decisive = wins + losses
    return {
        "n_samples": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_all": round(wins / n, 4) if n else 0.0,
        "win_rate_decisive": round(wins / decisive, 4) if decisive else 0.0,
    }


def _build_record(
    sample: EvalQuery,
    payload: dict[str, str],
    gt_field: str,
    judge_model: str | None,
) -> dict | None:
    gt_rating = payload.get("gt_rating")
    baseline_run_dir = payload.get("baseline_run_dir")
    full_run_dir = payload.get("full_run_dir")
    if not gt_rating or not baseline_run_dir or not full_run_dir:
        return None

    baseline_run_path = Path(baseline_run_dir)
    full_run_path = Path(full_run_dir)
    baseline_rating = _extract_generated_rating(baseline_run_path)
    full_rating = _extract_generated_rating(full_run_path)
    baseline_prediction_text = _extract_generated_prediction_text(baseline_run_path)
    full_prediction_text = _extract_generated_prediction_text(full_run_path)
    gt_prediction_text = _load_ground_truth_prediction_text(sample, gt_field)
    baseline_llm_judge = None
    full_llm_judge = None
    if judge_model:
        baseline_llm_judge = _judge_prediction_vs_gt(
            sample=sample,
            generated_label="baseline",
            generated_text=baseline_prediction_text,
            gt_text=gt_prediction_text,
            judge_model=judge_model,
        )
        full_llm_judge = _judge_prediction_vs_gt(
            sample=sample,
            generated_label="full",
            generated_text=full_prediction_text,
            gt_text=gt_prediction_text,
            judge_model=judge_model,
        )

    return {
        "report_id": sample.report_id,
        "gt_rating": gt_rating,
        "gt_prediction_text": gt_prediction_text,
        "baseline_rating": baseline_rating,
        "full_rating": full_rating,
        "baseline_match": baseline_rating == gt_rating,
        "full_match": full_rating == gt_rating,
        "baseline_prediction_text": baseline_prediction_text,
        "full_prediction_text": full_prediction_text,
        "baseline_llm_judge": baseline_llm_judge,
        "full_llm_judge": full_llm_judge,
        "baseline_run_dir": baseline_run_dir,
        "full_run_dir": full_run_dir,
    }


def main() -> None:
    args = parse_args()
    Config.validate_required()

    queries = load_queries(args.queries)
    selected = filter_queries(
        queries,
        report_ids=set(args.report_id) if args.report_id else None,
        limit=args.limit,
    )

    baseline_index = index_runs_by_query(args.baseline_runs_root)
    full_index = index_runs_by_query(args.full_runs_root)
    baseline_output_root = Path(args.baseline_output_root).resolve()
    full_output_root = Path(args.full_output_root).resolve()
    baseline_output_root.mkdir(parents=True, exist_ok=True)
    full_output_root.mkdir(parents=True, exist_ok=True)

    paired_samples = []
    for sample in selected:
        baseline_run_dir = baseline_index.get(sample.query)
        full_run_dir = full_index.get(sample.query)
        if baseline_run_dir and full_run_dir:
            paired_samples.append((sample, baseline_run_dir, full_run_dir))

    if not paired_samples:
        raise SystemExit("No paired baseline/full runs found for the selected samples.")

    sample_map = {sample.report_id: sample for sample, _, _ in paired_samples}
    future_map = {}
    rerun_results: dict[str, dict[str, str]] = {}
    total_jobs = len(paired_samples) * 2
    max_workers = max(1, min(args.max_workers, total_jobs))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sample, baseline_run_dir, full_run_dir in paired_samples:
            rerun_results[sample.report_id] = {"gt_rating": _normalize_rating(sample.rating)}
            jobs = [
                ("baseline", baseline_run_dir, baseline_output_root),
                ("full", full_run_dir, full_output_root),
            ]
            for mode, run_dir, output_root in jobs:
                print(f"[queued] {sample.report_id} mode={mode} source={run_dir}")
                future = executor.submit(
                    rerun_prediction_for_run,
                    Path(run_dir),
                    mode,
                    output_root,
                    f"{args.suffix}-{mode}",
                )
                future_map[future] = (sample.report_id, mode)

        completed = 0
        for future in as_completed(future_map):
            report_id, mode = future_map[future]
            try:
                new_run_dir = future.result()
                rerun_results[report_id][f"{mode}_run_dir"] = str(new_run_dir)
                completed += 1
                print(f"[done {completed}/{total_jobs}] {report_id} mode={mode} -> {new_run_dir.name}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rerun_results[report_id][f"{mode}_error"] = str(exc)
                completed += 1
                print(f"[error {completed}/{total_jobs}] {report_id} mode={mode} -> {exc}")

    records: list[dict] = []
    record_futures = {}
    record_workers = max(1, min(args.max_workers, len(rerun_results)))
    with ThreadPoolExecutor(max_workers=record_workers) as executor:
        for report_id, payload in rerun_results.items():
            future = executor.submit(
                _build_record,
                sample_map[report_id],
                payload,
                args.gt_field,
                args.judge_model,
            )
            record_futures[future] = report_id
        for future in as_completed(record_futures):
            try:
                record = future.result()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                report_id = record_futures[future]
                print(f"[record error] {report_id} -> {exc}")
                continue
            if record is not None:
                records.append(record)

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

    print(f"Saved rerun comparison results to {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
