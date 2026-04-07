"""
LLM-as-judge implementations for ERRGen evaluation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from textwrap import dedent

from evaluation.schemas import (
    EvalQuery,
    FactualJudgeOutput,
    PairwiseEvaluation,
    PairwiseJudgeOutput,
    QualityJudgeOutput,
    RunBundle,
    SectionPairwiseEvaluation,
    SingleReportEvaluation,
)
from evaluation.scoring import (
    aggregate_pairwise_winners,
    compute_final_label,
    compute_q_score,
    compute_tier,
    count_issue_findings,
    passes_accuracy_gate,
    resolve_pairwise_outcome,
)
from errgen.config import Config
from errgen.llm import build_messages, chat_json


@dataclass(frozen=True)
class _FactualEvaluationState:
    factual: FactualJudgeOutput
    error_counts: dict[str, int]
    severe_error_counts: dict[str, int]
    accuracy_gate_passed: bool


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


def _format_single_section(section, max_chars: int = 5000) -> str:
    lines: list[str] = []
    total = 0
    header = f"## {section.section_name}\n"
    lines.append(header)
    total += len(header)
    for idx, para in enumerate(section.paragraphs, start=1):
        entry = f"[P{idx}] {para.text.strip()}\n"
        if total + len(entry) > max_chars:
            lines.append("[...truncated...]\n")
            break
        lines.append(entry)
        total += len(entry)
    return "".join(lines)


def _chunk_source_date(metadata: dict) -> str:
    for key in ("date", "published_at", "publishedAt", "datetime", "time_published"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "N/A"


def _format_report_with_evidence(bundle: RunBundle, max_chars: int = 32000) -> str:
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
            para_header = (
                f"Paragraph {idx}:\n"
                f"{para.text.strip()}\n"
                f"Authoritative cited chunk IDs: {', '.join(para.chunk_ids) or 'none'}\n"
                f"Authoritative cited calc IDs: {', '.join(para.calc_ids) or 'none'}\n"
            )
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
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                entry = (
                    f"- [{chunk.chunk_id}] type={chunk.source_type.value}; "
                    f"source_date={_chunk_source_date(chunk.metadata)}; "
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


def _iter_section_findings(evaluation: SingleReportEvaluation, section_name: str):
    for category in (
        "unsupported_claims",
        "numerical_errors",
        "citation_mismatches",
        "temporal_violations",
        "inconsistencies",
    ):
        for finding in getattr(evaluation.issue_findings, category):
            if finding.section == section_name:
                yield category, finding


def _format_section_findings(evaluation: SingleReportEvaluation, section_name: str) -> str:
    findings = list(_iter_section_findings(evaluation, section_name))
    severe = [item for item in findings if item[1].severity == "severe"]
    minor = [item for item in findings if item[1].severity == "minor"]

    lines = [
        f"Section-specific findings: severe={len(severe)}, minor={len(minor)}",
    ]
    for category, finding in findings[:3]:
        detail = finding.offending_span or finding.explanation
        detail = detail.strip().replace("\n", " ")
        if len(detail) > 180:
            detail = detail[:180] + "..."
        lines.append(f"- {category} ({finding.severity}): {detail}")
    if not findings:
        lines.append("- none")
    return "\n".join(lines)


def _pair_sections(first_bundle: RunBundle, second_bundle: RunBundle):
    second_by_name = {
        section.section_name: section for section in second_bundle.report.sections
    }
    paired = []
    used_second: set[str] = set()
    for section in first_bundle.report.sections:
        match = second_by_name.get(section.section_name)
        if not match:
            continue
        paired.append((section.section_name, section, match))
        used_second.add(section.section_name)

    unmatched_first = [
        section for section in first_bundle.report.sections
        if section.section_name not in {name for name, _, _ in paired}
    ]
    unmatched_second = [
        section for section in second_bundle.report.sections
        if section.section_name not in used_second
    ]
    for first_section, second_section in zip(unmatched_first, unmatched_second):
        if first_section.section_name == second_section.section_name:
            section_name = first_section.section_name
        else:
            section_name = f"{first_section.section_name} / {second_section.section_name}"
        paired.append((section_name, first_section, second_section))
    return paired


def _factual_report_system_prompt() -> str:
    return dedent(
        """
        You are evaluating an equity research report for a controlled benchmark.

        Score strictly the Accuracy layer only:
        - Claim Support
        - Numerical Accuracy
        - Citation Precision
        - Temporal Validity
        - Consistency

        Scoring rules:
        - Every metric score must be an integer from 1 to 5.
        - "severe" means material enough to break the Accuracy Gate:
          a wrong important number, unsupported material conclusion, post-as-of claim,
          or contradiction that changes the investment interpretation.
        - "minor" means noticeable but not gate-breaking.
        - Use only the provided report text and evidence pack.
        - Be strict on financial claims and dates.
        - The paragraph text may contain inline citation labels such as C018 or K003.
          These inline labels are not authoritative identifiers for this evaluation.
          The authoritative support for each paragraph is the structured
          "Authoritative cited chunk IDs" and "Authoritative cited calc IDs" listed
          under that paragraph together with the evidence/calculation blocks below.
        - Do not mark a citation mismatch solely because an inline citation label in
          the prose does not literally match a raw UUID-like chunk ID shown below.
        - Do not mark a claim unsupported solely because an evidence snippet is
          truncated. Only mark unsupported or citation mismatch when the visible
          evidence clearly fails to support the claim, clearly contradicts it, or
          the cited structured materials are plainly insufficient.
        - For numerical claims, trust the provided calculation objects when they are
          directly relevant and internally consistent with the visible source data.
        - For temporal validity, rely on explicit source dates / statement dates /
          periods shown in the evidence pack. Only mark a temporal violation when
          post-as-of usage is clear.

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
          }
        }
        """
    ).strip()


def _quality_report_system_prompt() -> str:
    return dedent(
        """
        You are evaluating an equity research report for a controlled benchmark.

        The report has already passed the factual Accuracy Gate.
        Score strictly the Quality layer only:
        - Financial Numeric
        - News
        - CMI (Company, Market, Industry)
        - Invest
        - Risk
        - Writing

        Scoring rules:
        - Every metric score must be an integer from 1 to 5.
        - Judge quality, depth, completeness, and reasoning quality, not factual validity.
        - Use the report text only.
        - Reward reports that fully answer the user query with a complete investor-useful
          package: operating performance, catalysts, company/industry context,
          investment view, and risk implications.
        - Reward synthesis across sections. A strong report does not just list facts;
          it connects numbers, news, company positioning, risks, and recommendation logic.
        - Reward quantified analysis and explicit implications for investors when present.
        - Do not reward brevity by itself. A longer report should score higher when the
          extra content adds relevant analysis, context, or decision-useful detail.
        - For News, credit relevant recent developments even when they are integrated
          into business, risk, or investment sections rather than isolated in a single
          dedicated news section.
        - Do not over-penalize light reuse of anchor metrics across sections when each
          section uses them for a different analytical purpose. Penalize repetition only
          when it is near-verbatim, mostly redundant, and displaces new insight.
        - Do not heavily penalize a recommendation solely because it does not contain a
          full valuation model. If the recommendation is clear, logically supported by
          prior analysis, and highlights key catalysts / risks, it can still score well.
        - Treat transparent caveats about evidence limits as a positive sign of analytical
          discipline unless they prevent the report from providing a usable conclusion.
        - For Writing, value structure, clarity, and analytical flow over terseness.
          Do not double-penalize Writing for weaknesses already captured under Financial
          Numeric, CMI, Invest, or Risk unless they directly harm readability or structure.

        Return JSON only with this exact structure:
        {
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

        Important evaluation note:
        - Inline citation labels that appear inside the report prose are writer-side
          labels only.
        - Use the structured cited chunk IDs / calc IDs and the evidence blocks under
          each paragraph as the authoritative grounding record.

        === REPORT WITH EVIDENCE PACK ===
        {_format_report_with_evidence(bundle)}
        """
    ).strip()


def _quality_report_user_prompt(
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

        === REPORT ===
        {_format_report_sections(bundle)}
        """
    ).strip()


def _pairwise_system_prompt() -> str:
    return dedent(
        """
        You are comparing the corresponding section from two equity research reports
        for the same sample and the same user query.

        Decision policy:
        - Judge this section in the context of the full report's task, not as an
          isolated writing sample.
        - The user query is the primary objective. Prefer the section that better helps
          the full report answer the query.
        - Reliability for this section comes first when the gap is material.
        - Use the section-specific finding summaries as the main reliability signal.
          Do not let an unrelated issue elsewhere in the report dominate this section.
        - Treat report-level scores and report-level summaries as secondary context only.
          Use the section text as the primary basis for judgment.
        - Do not mechanically follow the higher overall score. If the section text
          clearly favors one report, prefer it even when its report-level score is lower.
        - If the section texts are close, you may use the overall scores and summaries
          as tie-breakers.
        - If reliability looks similar, decide on section quality.
        - Prefer the section with stronger depth, relevance to the user query,
          clearer analytical implications, and better organization.
        - Do not reward brevity or omission of analysis by itself.
        - Do not prefer a section merely because it is longer or shorter. Prefer the
          one with better signal density and investor usefulness.
        - Do not prefer a section merely because it sounds more polished, more
          templated, or more like sell-side house style.
        - If both sections share similar gaps, prefer the one that leaves the reader
          with a clearer takeaway rather than the one that merely sounds more polished
          or more templated.
        - If the two sections are materially indistinguishable, return tie.

        Return JSON only:
        {
          "factual_comparison": "...",
          "quality_comparison": "...",
          "winner": "first_report",
          "rationale": "..."
        }
        """
    ).strip()


def _format_quality_breakdown(evaluation: SingleReportEvaluation) -> str:
    quality = evaluation.quality
    return (
        f"financial_numeric={quality.financial_numeric.score}, "
        f"news={quality.news.score}, "
        f"cmi={quality.cmi.score}, "
        f"invest={quality.invest.score}, "
        f"risk={quality.risk.score}, "
        f"writing={quality.writing.score}"
    )


def _section_importance(section_name: str) -> str:
    importance_map = {
        "Company Overview": "low",
        "Recent Developments": "medium",
        "Financial Analysis": "high",
        "Business & Competitive Analysis": "medium",
        "Risk Analysis": "medium",
        "Investment Recommendation & Outlook": "high",
    }
    return importance_map.get(section_name, "medium")


def _pairwise_user_prompt(
    sample: EvalQuery,
    section_name: str,
    first_model: str,
    second_model: str,
    first_eval: SingleReportEvaluation,
    second_eval: SingleReportEvaluation,
    first_section,
    second_section,
) -> str:
    return dedent(
        f"""
        Sample ID: {sample.report_id}
        Ticker: {sample.ticker}
        As-of date: {sample.as_of_date}
        User query: {sample.query}
        Section: {section_name}
        Section importance: {_section_importance(section_name)}

        === FIRST REPORT: {first_model} ===
        Report-level accuracy gate: {first_eval.accuracy_gate_passed}
        Report-level Q score: {first_eval.q_score}
        Report-level quality summary: {_format_quality_breakdown(first_eval)}
        Report-level strength: {first_eval.summary.primary_strength}
        Report-level weakness: {first_eval.summary.primary_weakness}
        {_format_section_findings(first_eval, first_section.section_name)}
        Section text:
        {_format_single_section(first_section)}

        === SECOND REPORT: {second_model} ===
        Report-level accuracy gate: {second_eval.accuracy_gate_passed}
        Report-level Q score: {second_eval.q_score}
        Report-level quality summary: {_format_quality_breakdown(second_eval)}
        Report-level strength: {second_eval.summary.primary_strength}
        Report-level weakness: {second_eval.summary.primary_weakness}
        {_format_section_findings(second_eval, second_section.section_name)}
        Section text:
        {_format_single_section(second_section)}
        """
    ).strip()


def _skipped_quality_scores(reason: str) -> QualityJudgeOutput:
    return QualityJudgeOutput(
        quality={
            "financial_numeric": {"score": 1, "reason": reason},
            "news": {"score": 1, "reason": reason},
            "cmi": {"score": 1, "reason": reason},
            "invest": {"score": 1, "reason": reason},
            "risk": {"score": 1, "reason": reason},
            "writing": {"score": 1, "reason": reason},
        },
        summary={
            "primary_strength": "Quality evaluation skipped because the report failed the Accuracy Gate.",
            "primary_weakness": reason,
        },
    )


def _failure_reason_from_severe_counts(severe_counts: dict[str, int]) -> str:
    failing = [
        label.replace("_", " ")
        for label in (
            "unsupported_claims",
            "numerical_errors",
            "temporal_violations",
            "inconsistencies",
        )
        if severe_counts.get(label, 0) > 0
    ]
    if not failing:
        return "Skipped because the report did not pass the Accuracy Gate."
    labels = ", ".join(failing)
    return f"Skipped because the report failed the Accuracy Gate due to severe {labels}."


def _aggregate_order_result(
    section_results: list[SectionPairwiseEvaluation],
    first_model: str,
    second_model: str,
    order_attr: str,
) -> PairwiseJudgeOutput:
    first_sections: list[str] = []
    second_sections: list[str] = []
    tie_sections: list[str] = []
    for item in section_results:
        winner = getattr(item, order_attr).winner
        if winner == "first_report":
            first_sections.append(item.section_name)
        elif winner == "second_report":
            second_sections.append(item.section_name)
        else:
            tie_sections.append(item.section_name)

    if len(first_sections) > len(second_sections):
        overall_winner = "first_report"
    elif len(second_sections) > len(first_sections):
        overall_winner = "second_report"
    else:
        overall_winner = "tie"

    factual = (
        f"Across {len(section_results)} matched sections, {first_model} won "
        f"{len(first_sections)} sections on this ordering, {second_model} won "
        f"{len(second_sections)}, and {len(tie_sections)} were ties."
    )
    quality = (
        f"{first_model} stronger sections: {', '.join(first_sections) or 'none'}. "
        f"{second_model} stronger sections: {', '.join(second_sections) or 'none'}. "
        f"Ties: {', '.join(tie_sections) or 'none'}."
    )
    if overall_winner == "first_report":
        rationale = (
            f"Section-level vote favors {first_model} on this ordering."
        )
    elif overall_winner == "second_report":
        rationale = (
            f"Section-level vote favors {second_model} on this ordering."
        )
    else:
        rationale = "Section-level vote is tied on this ordering."

    return PairwiseJudgeOutput(
        factual_comparison=factual,
        quality_comparison=quality,
        winner=overall_winner,
        rationale=rationale,
    )


class LLMReportJudge:
    def __init__(self, judge_model: str | None = None) -> None:
        self.judge_model = judge_model or Config.OPENAI_MODEL

    def judge_factual(
        self,
        bundle: RunBundle,
        sample: EvalQuery,
        model_name: str,
    ) -> _FactualEvaluationState:
        messages = build_messages(
            _factual_report_system_prompt(),
            _single_report_user_prompt(bundle, sample, model_name),
        )
        raw = chat_json(messages, model=self.judge_model, temperature=0.0)
        factual = FactualJudgeOutput.model_validate(raw)

        error_counts, severe_error_counts = count_issue_findings(factual.issue_findings)
        accuracy_gate_passed = passes_accuracy_gate(severe_error_counts)
        return _FactualEvaluationState(
            factual=factual,
            error_counts=error_counts,
            severe_error_counts=severe_error_counts,
            accuracy_gate_passed=accuracy_gate_passed,
        )

    def judge_quality(
        self,
        bundle: RunBundle,
        sample: EvalQuery,
        model_name: str,
    ) -> QualityJudgeOutput:
        quality_messages = build_messages(
            _quality_report_system_prompt(),
            _quality_report_user_prompt(bundle, sample, model_name),
        )
        quality_raw = chat_json(
            quality_messages,
            model=self.judge_model,
            temperature=0.0,
        )
        return QualityJudgeOutput.model_validate(quality_raw)

    def build_single_evaluation(
        self,
        bundle: RunBundle,
        sample: EvalQuery,
        model_name: str,
        factual_state: _FactualEvaluationState,
        quality_result: QualityJudgeOutput | None = None,
    ) -> SingleReportEvaluation:
        if factual_state.accuracy_gate_passed:
            if quality_result is None:
                raise ValueError(
                    f"Quality result is required for passing report {sample.report_id} ({model_name})"
                )
            q_score = compute_q_score(quality_result.quality)
        else:
            quality_result = _skipped_quality_scores(
                _failure_reason_from_severe_counts(factual_state.severe_error_counts)
            )
            q_score = 0.0

        tier = compute_tier(factual_state.accuracy_gate_passed, q_score)
        final_label = compute_final_label(factual_state.accuracy_gate_passed, q_score)

        return SingleReportEvaluation(
            sample_id=sample.report_id,
            model_name=model_name,
            ticker=sample.ticker,
            as_of_date=sample.as_of_date,
            run_id=bundle.run_id,
            judge_model=self.judge_model,
            accuracy_gate_passed=factual_state.accuracy_gate_passed,
            accuracy=factual_state.factual.accuracy,
            issue_findings=factual_state.factual.issue_findings,
            error_counts=factual_state.error_counts,
            severe_error_counts=factual_state.severe_error_counts,
            quality=quality_result.quality,
            q_score=q_score,
            tier=tier,
            final_label=final_label,
            summary=quality_result.summary,
        )

    def judge_single(
        self,
        bundle: RunBundle,
        sample: EvalQuery,
        model_name: str,
    ) -> SingleReportEvaluation:
        factual_state = self.judge_factual(bundle, sample, model_name)
        quality_result = None
        if factual_state.accuracy_gate_passed:
            quality_result = self.judge_quality(bundle, sample, model_name)
        return self.build_single_evaluation(
            bundle=bundle,
            sample=sample,
            model_name=model_name,
            factual_state=factual_state,
            quality_result=quality_result,
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
        section_pairs = _pair_sections(bundle_a, bundle_b)
        if not section_pairs:
            raise ValueError(f"No comparable sections found for sample {sample.report_id}")

        max_workers = max(
            1,
            min(len(section_pairs), Config.EVAL_PAIRWISE_MAX_CONCURRENCY),
        )
        indexed_results: dict[int, SectionPairwiseEvaluation] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._judge_section_pairwise,
                    sample,
                    section_name,
                    model_a,
                    model_b,
                    eval_a,
                    eval_b,
                    first_section,
                    second_section,
                ): idx
                for idx, (section_name, first_section, second_section) in enumerate(section_pairs)
            }
            for future in as_completed(future_map):
                indexed_results[future_map[future]] = future.result()

        section_results = [
            indexed_results[idx] for idx in range(len(section_pairs))
        ]
        order_ab = _aggregate_order_result(section_results, model_a, model_b, "order_ab")
        order_ba = _aggregate_order_result(section_results, model_b, model_a, "order_ba")
        order_ab_winner = _map_pairwise_winner(order_ab.winner, model_a, model_b)
        order_ba_winner = _map_pairwise_winner(order_ba.winner, model_b, model_a)
        final_outcomes = [item.final_outcome for item in section_results]
        final_outcome = aggregate_pairwise_winners(final_outcomes, model_a, model_b)
        section_outcome_counts = {
            model_a: sum(1 for outcome in final_outcomes if outcome == model_a),
            model_b: sum(1 for outcome in final_outcomes if outcome == model_b),
            "Tie": sum(1 for outcome in final_outcomes if outcome == "Tie"),
        }

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
            section_results=section_results,
            section_outcome_counts=section_outcome_counts,
        )

    def _judge_section_pairwise(
        self,
        sample: EvalQuery,
        section_name: str,
        model_a: str,
        model_b: str,
        eval_a: SingleReportEvaluation,
        eval_b: SingleReportEvaluation,
        first_section,
        second_section,
    ) -> SectionPairwiseEvaluation:
        order_ab = self._judge_pairwise_once(
            sample=sample,
            section_name=section_name,
            first_model=model_a,
            second_model=model_b,
            first_eval=eval_a,
            second_eval=eval_b,
            first_section=first_section,
            second_section=second_section,
        )
        order_ba = self._judge_pairwise_once(
            sample=sample,
            section_name=section_name,
            first_model=model_b,
            second_model=model_a,
            first_eval=eval_b,
            second_eval=eval_a,
            first_section=second_section,
            second_section=first_section,
        )

        order_ab_winner = _map_pairwise_winner(order_ab.winner, model_a, model_b)
        order_ba_winner = _map_pairwise_winner(order_ba.winner, model_b, model_a)
        final_outcome = resolve_pairwise_outcome(order_ab_winner, order_ba_winner)

        return SectionPairwiseEvaluation(
            section_name=section_name,
            order_ab_winner=order_ab_winner,
            order_ba_winner=order_ba_winner,
            final_outcome=final_outcome,
            order_ab=order_ab,
            order_ba=order_ba,
        )

    def _judge_pairwise_once(
        self,
        sample: EvalQuery,
        section_name: str,
        first_model: str,
        second_model: str,
        first_eval: SingleReportEvaluation,
        second_eval: SingleReportEvaluation,
        first_section,
        second_section,
    ) -> PairwiseJudgeOutput:
        messages = build_messages(
            _pairwise_system_prompt(),
            _pairwise_user_prompt(
                sample=sample,
                section_name=section_name,
                first_model=first_model,
                second_model=second_model,
                first_eval=first_eval,
                second_eval=second_eval,
                first_section=first_section,
                second_section=second_section,
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
