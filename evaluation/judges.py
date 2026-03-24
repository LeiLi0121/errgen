"""
LLM-as-judge implementations for ERRGen evaluation.
"""

from __future__ import annotations

from textwrap import dedent

from evaluation.schemas import (
    EvalQuery,
    PairwiseEvaluation,
    PairwiseJudgeOutput,
    RunBundle,
    SingleReportEvaluation,
    SingleReportJudgeOutput,
)
from evaluation.scoring import (
    compute_final_label,
    compute_q_score,
    compute_tier,
    count_issue_findings,
    passes_accuracy_gate,
    resolve_pairwise_outcome,
)
from errgen.config import Config
from errgen.llm import build_messages, chat_json


def _format_report_sections(bundle: RunBundle, max_chars: int = 12000) -> str:
    lines: list[str] = []
    total = 0
    for section in bundle.report.sections:
        header = f"## {section.section_name}\n"
        if total + len(header) > max_chars:
            break
        lines.append(header)
        total += len(header)
        for idx, para in enumerate(section.paragraphs, start=1):
            entry = f"[P{idx}] {para.text.strip()}\n"
            if total + len(entry) > max_chars:
                lines.append("[...truncated...]\n")
                return "".join(lines)
            lines.append(entry)
            total += len(entry)
        lines.append("\n")
        total += 1
    return "".join(lines)


def _format_report_with_evidence(bundle: RunBundle, max_chars: int = 22000) -> str:
    chunk_map = {chunk.chunk_id: chunk for chunk in bundle.evidence_chunks}
    calc_map = {calc.calc_id: calc for calc in bundle.calculation_results}

    lines: list[str] = []
    total = 0
    for section in bundle.report.sections:
        section_header = f"## {section.section_name}\n"
        if total + len(section_header) > max_chars:
            break
        lines.append(section_header)
        total += len(section_header)

        for idx, para in enumerate(section.paragraphs, start=1):
            para_header = f"Paragraph {idx}:\n{para.text.strip()}\n"
            if total + len(para_header) > max_chars:
                lines.append("[...truncated...]\n")
                return "".join(lines)
            lines.append(para_header)
            total += len(para_header)

            if para.chunk_ids:
                lines.append("Cited evidence:\n")
                total += len("Cited evidence:\n")
            for chunk_id in para.chunk_ids:
                chunk = chunk_map.get(chunk_id)
                if not chunk:
                    continue
                snippet = chunk.text.strip().replace("\n", " ")
                if len(snippet) > 220:
                    snippet = snippet[:220] + "..."
                entry = (
                    f"- [{chunk.chunk_id}] type={chunk.source_type.value}; "
                    f"period={chunk.period or 'N/A'}; field={chunk.field_name or 'N/A'}; "
                    f"evidence={snippet}\n"
                )
                if total + len(entry) > max_chars:
                    lines.append("[...truncated...]\n")
                    return "".join(lines)
                lines.append(entry)
                total += len(entry)

            if para.calc_ids:
                lines.append("Cited calculations:\n")
                total += len("Cited calculations:\n")
            for calc_id in para.calc_ids:
                calc = calc_map.get(calc_id)
                if not calc:
                    continue
                entry = (
                    f"- [{calc.calc_id}] {calc.description}: {calc.result} "
                    f"(error={calc.error or 'none'})\n"
                )
                if total + len(entry) > max_chars:
                    lines.append("[...truncated...]\n")
                    return "".join(lines)
                lines.append(entry)
                total += len(entry)

            lines.append("\n")
            total += 1
    return "".join(lines)


def _single_report_system_prompt() -> str:
    return dedent(
        """
        You are evaluating an equity research report for a controlled benchmark.

        Score strictly according to this rubric:
        1. Accuracy layer
           - Claim Support
           - Numerical Accuracy
           - Citation Precision
           - Temporal Validity
           - Consistency
        2. Quality layer
           - Financial Numeric
           - News
           - CMI (Company, Market, Industry)
           - Invest
           - Risk
           - Writing

        Scoring rules:
        - Every metric score must be an integer from 1 to 5.
        - "severe" means material enough to break the Accuracy Gate:
          a wrong important number, unsupported material conclusion, post-as-of claim,
          or contradiction that changes the investment interpretation.
        - "minor" means noticeable but not gate-breaking.
        - Use only the provided report text and evidence pack.
        - Be strict on financial claims and dates.

        Return JSON only with this exact structure:
        {
          "accuracy": {
            "claim_support": {"score": 1, "reason": "..."},
            "numerical_accuracy": {"score": 1, "reason": "..."},
            "citation_precision": {"score": 1, "reason": "..."},
            "temporal_validity": {"score": 1, "reason": "..."},
            "consistency": {"score": 1, "reason": "..."}
          },
          "issue_findings": {
            "unsupported_claims": [{"severity": "severe", "explanation": "...", "section": "...", "offending_span": "..."}],
            "numerical_errors": [{"severity": "severe", "explanation": "...", "section": "...", "offending_span": "..."}],
            "citation_mismatches": [{"severity": "minor", "explanation": "...", "section": "...", "offending_span": "..."}],
            "temporal_violations": [{"severity": "severe", "explanation": "...", "section": "...", "offending_span": "..."}],
            "inconsistencies": [{"severity": "severe", "explanation": "...", "section": "...", "offending_span": "..."}]
          },
          "quality": {
            "financial_numeric": {"score": 1, "reason": "..."},
            "news": {"score": 1, "reason": "..."},
            "cmi": {"score": 1, "reason": "..."},
            "invest": {"score": 1, "reason": "..."},
            "risk": {"score": 1, "reason": "..."},
            "writing": {"score": 1, "reason": "..."}
          },
          "summary": {
            "primary_strength": "...",
            "primary_weakness": "..."
          }
        }
        """
    ).strip()


def _single_report_user_prompt(
    bundle: RunBundle,
    sample: EvalQuery,
    model_name: str,
) -> str:
    return dedent(
        f"""
        Sample ID: {sample.report_id}
        Model: {model_name}
        Ticker: {sample.ticker}
        As-of date: {sample.as_of_date}
        Query: {sample.query}

        === REPORT WITH EVIDENCE PACK ===
        {_format_report_with_evidence(bundle)}
        """
    ).strip()


def _pairwise_system_prompt() -> str:
    return dedent(
        """
        You are comparing two equity research reports for the same sample.

        Decision policy:
        - Reliability first: prefer the report with fewer factual and temporal problems.
        - Then compare the six quality dimensions.
        - If the two reports are materially indistinguishable overall, return tie.

        Return JSON only:
        {
          "factual_comparison": "...",
          "quality_comparison": "...",
          "winner": "first_report",
          "rationale": "..."
        }
        """
    ).strip()


def _pairwise_user_prompt(
    sample: EvalQuery,
    first_model: str,
    second_model: str,
    first_eval: SingleReportEvaluation,
    second_eval: SingleReportEvaluation,
    first_bundle: RunBundle,
    second_bundle: RunBundle,
) -> str:
    return dedent(
        f"""
        Sample ID: {sample.report_id}
        Ticker: {sample.ticker}
        As-of date: {sample.as_of_date}

        === FIRST REPORT: {first_model} ===
        Accuracy gate: {first_eval.accuracy_gate_passed}
        Q score: {first_eval.q_score}
        Tier: {first_eval.tier}
        Severe errors: {first_eval.severe_error_counts}
        Strength: {first_eval.summary.primary_strength}
        Weakness: {first_eval.summary.primary_weakness}
        Report:
        {_format_report_sections(first_bundle)}

        === SECOND REPORT: {second_model} ===
        Accuracy gate: {second_eval.accuracy_gate_passed}
        Q score: {second_eval.q_score}
        Tier: {second_eval.tier}
        Severe errors: {second_eval.severe_error_counts}
        Strength: {second_eval.summary.primary_strength}
        Weakness: {second_eval.summary.primary_weakness}
        Report:
        {_format_report_sections(second_bundle)}
        """
    ).strip()


class LLMReportJudge:
    def __init__(self, judge_model: str | None = None) -> None:
        self.judge_model = judge_model or Config.OPENAI_MODEL

    def judge_single(
        self,
        bundle: RunBundle,
        sample: EvalQuery,
        model_name: str,
    ) -> SingleReportEvaluation:
        messages = build_messages(
            _single_report_system_prompt(),
            _single_report_user_prompt(bundle, sample, model_name),
        )
        raw = chat_json(messages, model=self.judge_model, temperature=0.0)
        parsed = SingleReportJudgeOutput.model_validate(raw)

        error_counts, severe_error_counts = count_issue_findings(parsed.issue_findings)
        accuracy_gate_passed = passes_accuracy_gate(severe_error_counts)
        q_score = compute_q_score(parsed.quality)
        tier = compute_tier(accuracy_gate_passed, q_score)
        final_label = compute_final_label(accuracy_gate_passed, q_score)

        return SingleReportEvaluation(
            sample_id=sample.report_id,
            model_name=model_name,
            ticker=sample.ticker,
            as_of_date=sample.as_of_date,
            run_id=bundle.run_id,
            judge_model=self.judge_model,
            accuracy_gate_passed=accuracy_gate_passed,
            accuracy=parsed.accuracy,
            error_counts=error_counts,
            severe_error_counts=severe_error_counts,
            quality=parsed.quality,
            q_score=q_score,
            tier=tier,
            final_label=final_label,
            summary=parsed.summary,
        )

    def judge_pairwise(
        self,
        sample: EvalQuery,
        model_a: str,
        model_b: str,
        eval_a: SingleReportEvaluation,
        eval_b: SingleReportEvaluation,
        bundle_a: RunBundle,
        bundle_b: RunBundle,
    ) -> PairwiseEvaluation:
        order_ab = self._judge_pairwise_once(
            sample=sample,
            first_model=model_a,
            second_model=model_b,
            first_eval=eval_a,
            second_eval=eval_b,
            first_bundle=bundle_a,
            second_bundle=bundle_b,
        )
        order_ba = self._judge_pairwise_once(
            sample=sample,
            first_model=model_b,
            second_model=model_a,
            first_eval=eval_b,
            second_eval=eval_a,
            first_bundle=bundle_b,
            second_bundle=bundle_a,
        )

        order_ab_winner = _map_pairwise_winner(order_ab.winner, model_a, model_b)
        order_ba_winner = _map_pairwise_winner(order_ba.winner, model_b, model_a)
        final_outcome = resolve_pairwise_outcome(order_ab_winner, order_ba_winner)

        return PairwiseEvaluation(
            sample_id=sample.report_id,
            ticker=sample.ticker,
            as_of_date=sample.as_of_date,
            model_a=model_a,
            model_b=model_b,
            judge_model=self.judge_model,
            order_ab_winner=order_ab_winner,
            order_ba_winner=order_ba_winner,
            final_outcome=final_outcome,
            order_ab=order_ab,
            order_ba=order_ba,
        )

    def _judge_pairwise_once(
        self,
        sample: EvalQuery,
        first_model: str,
        second_model: str,
        first_eval: SingleReportEvaluation,
        second_eval: SingleReportEvaluation,
        first_bundle: RunBundle,
        second_bundle: RunBundle,
    ) -> PairwiseJudgeOutput:
        messages = build_messages(
            _pairwise_system_prompt(),
            _pairwise_user_prompt(
                sample=sample,
                first_model=first_model,
                second_model=second_model,
                first_eval=first_eval,
                second_eval=second_eval,
                first_bundle=first_bundle,
                second_bundle=second_bundle,
            ),
        )
        raw = chat_json(messages, model=self.judge_model, temperature=0.0)
        return PairwiseJudgeOutput.model_validate(raw)


def _map_pairwise_winner(winner: str, first_model: str, second_model: str) -> str:
    if winner == "first_report":
        return first_model
    if winner == "second_report":
        return second_model
    return "Tie"
