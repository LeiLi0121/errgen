"""
Evaluate ERRGen full reports against ground-truth reports with a quality-focused
pairwise LLM judge.

Typical usage:

python scripts/evaluate_full_vs_ground_truth.py \
  --queries evaluation/reports_extracted/errgen_queries.jsonl \
  --full-runs-root evaluation/runs/full \
  --output-dir evaluation/results_full_vs_gt \
  --gt-field narrative_text_path \
  --judge-model qwen3.5-plus
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from textwrap import dedent

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from pydantic import BaseModel

from evaluation.io import filter_queries, index_runs_by_query, load_queries, load_run_bundle
from evaluation.schemas import EvalQuery
from errgen.config import Config
from errgen.llm import build_messages, chat_json


class QualityPairwiseJudgeOutput(BaseModel):
    coverage_comparison: str
    analytical_depth_comparison: str
    writing_comparison: str
    winner: str
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full ERRGen reports against ground-truth reports with a quality-focused pairwise judge.",
    )
    parser.add_argument(
        "--queries",
        default="evaluation/reports_extracted/errgen_queries.jsonl",
        help="Path to the evaluation query set.",
    )
    parser.add_argument("--full-runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--gt-field",
        default="narrative_text_path",
        choices=["narrative_text_path", "full_text_path"],
        help="Which ground-truth text field to use from the query jsonl.",
    )
    parser.add_argument(
        "--generated-label",
        default="full",
        help="Label for the generated report in outputs.",
    )
    parser.add_argument(
        "--ground-truth-label",
        default="ground_truth",
        help="Label for the ground-truth report in outputs.",
    )
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
        default=4,
        help="Maximum number of concurrent evaluation workers.",
    )
    return parser.parse_args()


def _pairwise_system_prompt() -> str:
    return dedent(
        """
        You are comparing two equity research reports for the same sample.

        Your task is quality-focused pairwise evaluation against a ground-truth style
        benchmark. Judge which report is the stronger equity research deliverable.

        Evaluate only quality dimensions:
        - Query coverage and completeness
        - Analytical depth and synthesis
        - Investor usefulness of the investment view
        - Risk discussion quality
        - Writing quality and report structure

        Important rules:
        - Do not reward a report merely for being longer.
        - Do not reward generic verbosity, boilerplate, or repeated facts.
        - Prefer the report that better covers the requested dimensions with sharper,
          more decision-useful synthesis.
        - Ground truth is not automatically correct or automatically better. Judge the
          two texts on quality as written.
        - If one report is better in coverage but clearly weaker in synthesis and
          investor usefulness, you may still prefer the other.
        - If they are materially comparable overall, return tie.

        Return JSON only:
        {
          "coverage_comparison": "...",
          "analytical_depth_comparison": "...",
          "writing_comparison": "...",
          "winner": "first_report",
          "rationale": "..."
        }
        """
    ).strip()


def _truncate(text: str, max_chars: int = 16000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def _pairwise_user_prompt(
    sample: EvalQuery,
    first_label: str,
    second_label: str,
    first_report: str,
    second_report: str,
) -> str:
    headline = sample.headline or "N/A"
    rating = sample.rating or "N/A"
    target_price = (
        f"{sample.target_price} {sample.target_currency or ''}".strip()
        if sample.target_price is not None
        else "N/A"
    )
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

        Evaluate which report is higher quality for this task.

        === FIRST REPORT: {first_label} ===
        {_truncate(first_report)}

        === SECOND REPORT: {second_label} ===
        {_truncate(second_report)}
        """
    ).strip()


def _judge_once(
    sample: EvalQuery,
    first_label: str,
    second_label: str,
    first_report: str,
    second_report: str,
    judge_model: str,
) -> QualityPairwiseJudgeOutput:
    messages = build_messages(
        _pairwise_system_prompt(),
        _pairwise_user_prompt(
            sample=sample,
            first_label=first_label,
            second_label=second_label,
            first_report=first_report,
            second_report=second_report,
        ),
    )
    raw = chat_json(messages, model=judge_model, temperature=0.0)
    return QualityPairwiseJudgeOutput.model_validate(raw)


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


def _load_ground_truth_text(sample: EvalQuery, gt_field: str) -> str:
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
    return path.read_text(encoding="utf-8")


def _summarize(results: list[dict], preferred_label: str) -> dict:
    wins = sum(1 for item in results if item["final_outcome"] == preferred_label)
    losses = sum(
        1
        for item in results
        if item["final_outcome"] not in {preferred_label, "Tie"}
    )
    ties = sum(1 for item in results if item["final_outcome"] == "Tie")
    total = len(results)
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_all": round(wins / total, 4) if total else 0.0,
        "win_rate_decisive": round(wins / decisive, 4) if decisive else 0.0,
    }


def _evaluate_sample(
    sample: EvalQuery,
    full_run_dir,
    gt_field: str,
    judge_model: str,
    generated_label: str,
    ground_truth_label: str,
) -> dict:
    full_bundle = load_run_bundle(full_run_dir)
    gt_text = _load_ground_truth_text(sample, gt_field)

    order_ab = _judge_once(
        sample=sample,
        first_label=generated_label,
        second_label=ground_truth_label,
        first_report=full_bundle.report_text,
        second_report=gt_text,
        judge_model=judge_model,
    )
    order_ba = _judge_once(
        sample=sample,
        first_label=ground_truth_label,
        second_label=generated_label,
        first_report=gt_text,
        second_report=full_bundle.report_text,
        judge_model=judge_model,
    )

    order_ab_winner = _map_winner(
        order_ab.winner, generated_label, ground_truth_label
    )
    order_ba_winner = _map_winner(
        order_ba.winner, ground_truth_label, generated_label
    )
    final_outcome = _resolve_outcome(order_ab_winner, order_ba_winner)

    return {
        "sample_id": sample.report_id,
        "ticker": sample.ticker,
        "as_of_date": sample.as_of_date,
        "judge_model": judge_model,
        "generated_label": generated_label,
        "ground_truth_label": ground_truth_label,
        "ground_truth_field": gt_field,
        "full_run_id": full_bundle.run_id,
        "order_ab_winner": order_ab_winner,
        "order_ba_winner": order_ba_winner,
        "final_outcome": final_outcome,
        "order_ab": order_ab.model_dump(mode="json"),
        "order_ba": order_ba.model_dump(mode="json"),
    }


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)
    selected = filter_queries(
        queries,
        report_ids=set(args.report_id) if args.report_id else None,
        limit=args.limit,
    )

    full_index = index_runs_by_query(args.full_runs_root)
    judge_model = args.judge_model or Config.OPENAI_MODEL

    output_dir = Path(args.output_dir)
    pairwise_dir = output_dir / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(selected)
    max_workers = max(1, min(args.max_workers, total))
    future_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, sample in enumerate(selected, start=1):
            print(f"[{idx}/{total}] queued {sample.report_id} ({sample.ticker})")
            full_run_dir = full_index.get(sample.query)
            if not full_run_dir:
                raise FileNotFoundError(f"Missing full run for sample {sample.report_id}")
            future = executor.submit(
                _evaluate_sample,
                sample,
                full_run_dir,
                args.gt_field,
                judge_model,
                args.generated_label,
                args.ground_truth_label,
            )
            future_map[future] = sample.report_id

        completed = 0
        for future in as_completed(future_map):
            sample_id = future_map[future]
            result = future.result()
            completed += 1
            results.append(result)
            (pairwise_dir / f"{result['sample_id']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[done {completed}/{total}] saved: {sample_id}")

    summary = _summarize(results, preferred_label=args.generated_label)
    summary_payload = {
        "generated_label": args.generated_label,
        "ground_truth_label": args.ground_truth_label,
        "ground_truth_field": args.gt_field,
        "n_samples": len(results),
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved evaluation results to {output_dir}")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
