"""
Report assembler and Markdown renderer.

ReportAssembler takes the verified sections and constructs the FinalReport
data object. ReportRenderer converts a FinalReport to a Markdown string
suitable for saving or display.
"""

from __future__ import annotations

import logging
from datetime import datetime

from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    EvidenceChunk,
    FinalReport,
    IssueSeverity,
    ReportTable,
    ReportSection,
    SourceType,
    UserRequest,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.2f}"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def _extract_profile_value(text: str, key: str) -> str | None:
    marker = f"{key}:"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    return tail.split("|", 1)[0].strip()


def _period_sort_key(period: str) -> tuple[int, int]:
    parts = period.strip().split()
    if len(parts) < 2:
        return (0, 0)
    label, year_str = parts[0].upper(), parts[-1]
    try:
        year = int(year_str)
    except ValueError:
        return (0, 0)
    order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}.get(label, 0)
    return (year, order)


def _build_report_tables(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> list[ReportTable]:
    tables: list[ReportTable] = []
    for builder in (
        _company_snapshot_table,
        _market_performance_table,
        _key_financials_table,
        _key_ratio_table,
        _recent_filings_table,
    ):
        table = builder(request, all_chunks, calc_results)
        if table:
            tables.append(table)
    return tables


def _company_snapshot_table(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> ReportTable | None:
    del calc_results
    overview = next(
        (
            chunk for chunk in all_chunks
            if chunk.source_type == SourceType.COMPANY_PROFILE
            and chunk.field_name == "overview"
        ),
        None,
    )
    if not overview:
        return None
    rows = [
        ["Company", request.company_name or request.ticker],
        ["Ticker", request.ticker],
        ["Exchange", _extract_profile_value(overview.text, "Exchange") or "N/A"],
        ["Sector", _extract_profile_value(overview.text, "Sector") or "N/A"],
        ["Industry", _extract_profile_value(overview.text, "Industry") or "N/A"],
        ["Market Cap", _extract_profile_value(overview.text, "Market Cap") or "N/A"],
        ["Country", _extract_profile_value(overview.text, "Country") or "N/A"],
        ["Employees", _extract_profile_value(overview.text, "Full-Time Employees") or "N/A"],
    ]
    return ReportTable(
        title="Company Snapshot",
        columns=["Metric", "Value"],
        rows=rows,
        description="Deterministic company metadata pulled from the company profile source.",
    )


def _market_performance_table(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> ReportTable | None:
    del request
    stock_chunk = next(
        (
            chunk for chunk in all_chunks
            if chunk.source_type == SourceType.PRICE_DATA
            and chunk.metadata.get("series_role") == "stock"
        ),
        None,
    )
    benchmark_chunk = next(
        (
            chunk for chunk in all_chunks
            if chunk.source_type == SourceType.PRICE_DATA
            and chunk.metadata.get("series_role") == "benchmark"
        ),
        None,
    )
    if not stock_chunk:
        return None

    calc_map = {calc.description: calc for calc in calc_results if calc.error is None}
    benchmark_ticker = (
        benchmark_chunk.metadata.get("ticker")
        if benchmark_chunk
        else stock_chunk.metadata.get("benchmark_ticker", "N/A")
    )
    rows = [
        ["Benchmark", str(benchmark_ticker)],
        ["Company start price", _fmt_money(stock_chunk.metadata.get("price_start"))],
        ["Company end price", _fmt_money(stock_chunk.metadata.get("price_end"))],
        ["Company 52-week high", _fmt_money(stock_chunk.metadata.get("price_high"))],
        ["Company 52-week low", _fmt_money(stock_chunk.metadata.get("price_low"))],
    ]
    if benchmark_chunk:
        rows.extend(
            [
                ["Benchmark start price", _fmt_money(benchmark_chunk.metadata.get("price_start"))],
                ["Benchmark end price", _fmt_money(benchmark_chunk.metadata.get("price_end"))],
            ]
        )

    target_ticker = stock_chunk.metadata.get("ticker", "")
    rows.extend(
        [
            ["30-day company return", _fmt_pct(_calc_result_value(calc_map.get(f"{target_ticker} 30d price return")))],
            ["30-day benchmark return", _fmt_pct(_calc_result_value(calc_map.get(f"{benchmark_ticker} 30d benchmark return")))],
            ["30-day excess return", _fmt_pct(_calc_result_value(calc_map.get(f"{target_ticker} 30d excess return vs {benchmark_ticker}")))],
            ["1-year company return", _fmt_pct(_calc_result_value(calc_map.get(f"{target_ticker} 1y price return")))],
            ["1-year benchmark return", _fmt_pct(_calc_result_value(calc_map.get(f"{benchmark_ticker} 1y benchmark return")))],
            ["1-year excess return", _fmt_pct(_calc_result_value(calc_map.get(f"{target_ticker} 1y excess return vs {benchmark_ticker}")))],
        ]
    )
    return ReportTable(
        title="Market Performance",
        columns=["Metric", "Value"],
        rows=rows,
        description="Recent company-versus-benchmark price performance used by the prediction agent.",
    )


def _key_financials_table(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> ReportTable | None:
    del request, calc_results
    desired_fields = [
        ("Revenue", "revenue"),
        ("Gross Profit", "grossProfit"),
        ("Operating Income", "operatingIncome"),
        ("Net Income", "netIncome"),
        ("EPS", "eps"),
        ("Operating Cash Flow", "operatingCashFlow"),
        ("Free Cash Flow", "freeCashFlow"),
    ]
    chunk_map: dict[tuple[str, str], EvidenceChunk] = {}
    periods: set[str] = set()

    for chunk in all_chunks:
        if chunk.source_type not in {
            SourceType.INCOME_STATEMENT,
            SourceType.CASH_FLOW,
        }:
            continue
        if not chunk.period or not chunk.field_name or chunk.numeric_value is None:
            continue
        chunk_map[(chunk.period, chunk.field_name)] = chunk
        periods.add(chunk.period)

    ordered_periods = sorted(periods, key=_period_sort_key, reverse=True)[:4]
    if not ordered_periods:
        return None

    rows: list[list[str]] = []
    for label, field_name in desired_fields:
        row = [label]
        present = False
        for period in ordered_periods:
            chunk = chunk_map.get((period, field_name))
            if chunk:
                present = True
                if field_name == "eps":
                    row.append(_fmt_number(chunk.numeric_value))
                else:
                    row.append(_fmt_money(chunk.numeric_value))
            else:
                row.append("N/A")
        if present:
            rows.append(row)

    if not rows:
        return None

    return ReportTable(
        title="Key Financials",
        columns=["Metric"] + ordered_periods,
        rows=rows,
        description="Recent reported financial metrics sourced directly from structured statement data.",
    )


def _key_ratio_table(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> ReportTable | None:
    del request, all_chunks
    targets = [
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "EBITDA Margin",
        "R&D Intensity",
        "Current Ratio",
        "Debt-to-Equity",
        "Net Debt",
        "FCF Margin",
    ]
    rows: list[list[str]] = []
    for label in targets:
        calc = next(
            (
                item for item in calc_results
                if item.error is None and label in item.description
            ),
            None,
        )
        if not calc:
            continue
        if label in {"Current Ratio", "Debt-to-Equity"}:
            rendered = _fmt_number(_calc_result_value(calc))
        elif label == "Net Debt":
            rendered = _fmt_money(_calc_result_value(calc))
        else:
            rendered = _fmt_pct(_calc_result_value(calc))
        rows.append([label, rendered])

    if not rows:
        return None

    return ReportTable(
        title="Key Ratios",
        columns=["Metric", "Value"],
        rows=rows,
        description="Deterministic ratios computed from statement data and price signals.",
    )


def _recent_filings_table(
    request: UserRequest,
    all_chunks: list[EvidenceChunk],
    calc_results: list[CalculationResult],
) -> ReportTable | None:
    del request, calc_results
    rows: list[list[str]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in all_chunks:
        if chunk.source_type != SourceType.FILING or chunk.field_name != "filing_summary":
            continue
        form_type = str(chunk.metadata.get("form_type", "N/A"))
        filed_at = str(chunk.metadata.get("filed_at", "N/A"))
        title = str(chunk.metadata.get("filing_title", "N/A"))
        url = str(chunk.metadata.get("url", "N/A"))
        key = (form_type, filed_at, title)
        if key in seen:
            continue
        seen.add(key)
        rows.append([form_type, filed_at[:10], title[:60], url[:80]])

    if not rows:
        return None

    rows.sort(key=lambda row: row[1], reverse=True)
    return ReportTable(
        title="Recent Filings",
        columns=["Form", "Filed", "Title", "Source"],
        rows=rows[:5],
        description="Recent annual and quarterly filings made available to the analysis agents.",
    )


def _calc_result_value(calc: CalculationResult | None) -> float | None:
    if not calc or calc.error is not None:
        return None
    if isinstance(calc.result, (int, float)):
        return float(calc.result)
    return None


class ReportAssembler:
    """
    Assembles a FinalReport from verified sections, evidence, and calculations.

    Also determines the overall verification status:
    - PASS   if all sections passed
    - UNRESOLVED if any section has unresolved blocking issues
    - FAIL   otherwise
    """

    def assemble(
        self,
        request: UserRequest,
        sections: list[ReportSection],
        all_chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        warnings: list[str] | None = None,
    ) -> FinalReport:
        warnings = warnings or []

        # Collect only chunks that are actually cited in the final sections
        cited_chunk_ids: set[str] = set()
        cited_calc_ids: set[str] = set()
        for section in sections:
            for para in section.paragraphs:
                cited_chunk_ids.update(para.chunk_ids)
                cited_calc_ids.update(para.calc_ids)

        evidence_appendix = [c for c in all_chunks if c.chunk_id in cited_chunk_ids]
        calc_appendix = [c for c in calc_results if c.calc_id in cited_calc_ids]

        # Determine overall status
        overall = self._compute_overall_status(sections)

        if overall == VerificationStatus.UNRESOLVED:
            warnings.append(
                "Some paragraphs could not be fully verified within the allowed "
                "iteration limit. These are marked 'unresolved' in the report."
            )

        report = FinalReport(
            request=request,
            sections=sections,
            tables=_build_report_tables(request, all_chunks, calc_results),
            evidence_appendix=evidence_appendix,
            calculation_appendix=calc_appendix,
            overall_status=overall,
            warnings=warnings,
        )

        logger.info(
            "ReportAssembler: assembled report %s | %d sections | %d cited chunks | "
            "overall_status=%s",
            report.report_id,
            len(sections),
            len(evidence_appendix),
            overall.value,
        )
        return report

    @staticmethod
    def _compute_overall_status(sections: list[ReportSection]) -> VerificationStatus:
        if any(s.verification_status == VerificationStatus.UNRESOLVED for s in sections):
            return VerificationStatus.UNRESOLVED
        if all(s.verification_status == VerificationStatus.PASS for s in sections):
            return VerificationStatus.PASS
        return VerificationStatus.FAIL


class ReportRenderer:
    """
    Converts a FinalReport to a structured Markdown document.

    Each paragraph includes a citation footer linking to the evidence appendix.
    """

    def render(self, report: FinalReport) -> str:
        lines: list[str] = []

        # Header
        req = report.request
        lines.append(f"# Equity Research Report: {req.ticker}")
        if req.company_name:
            lines.append(f"**{req.company_name}** ({req.ticker})")
        lines.append(f"**As-of Date:** {req.as_of_date or 'Not specified'}")
        lines.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(
            f"**Report ID:** `{report.report_id}`  |  "
            f"**Verification Status:** `{report.overall_status.value.upper()}`"
        )
        if req.focus_areas:
            lines.append(f"**Focus Areas:** {', '.join(req.focus_areas)}")
        lines.append("")

        # Warnings banner
        if report.warnings:
            lines.append("> **⚠ Warnings**")
            for w in report.warnings:
                lines.append(f"> - {w}")
            lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        toc_index = 1
        if report.tables:
            for table in report.tables:
                anchor = table.title.lower().replace(" ", "-")
                anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
                lines.append(f"{toc_index}. [{table.title}](#{anchor})")
                toc_index += 1
        for section in report.sections:
            anchor = section.section_name.lower().replace(" ", "-").replace("&", "")
            anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
            status_badge = self._status_badge(section.verification_status)
            lines.append(f"{toc_index}. [{section.section_name}](#{anchor}) {status_badge}")
            toc_index += 1
        lines.append(f"{toc_index}. [Evidence Appendix](#evidence-appendix)")
        lines.append(f"{toc_index + 1}. [Calculation Appendix](#calculation-appendix)")
        lines.append("")

        if report.tables:
            for table in report.tables:
                lines.extend(self._render_table(table))

        # Sections
        for section in report.sections:
            lines.extend(self._render_section(section))

        # Evidence appendix
        lines.append("---")
        lines.append("## Evidence Appendix")
        lines.append(
            f"*{len(report.evidence_appendix)} evidence chunks cited in this report.*"
        )
        lines.append("")
        for chunk in report.evidence_appendix:
            lines.extend(self._render_evidence_chunk(chunk))

        # Calculation appendix
        lines.append("---")
        lines.append("## Calculation Appendix")
        lines.append(
            f"*{len(report.calculation_appendix)} deterministic calculations.*"
        )
        lines.append("")
        for calc in report.calculation_appendix:
            lines.extend(self._render_calc(calc))

        lines.append("---")
        lines.append(
            "*This report was generated by the ERRGen evidence-grounded pipeline. "
            "All analytical claims are traceable to cited evidence chunks and "
            "deterministic calculations listed in the appendices above.*"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Section rendering
    # ------------------------------------------------------------------

    def _render_section(self, section: ReportSection) -> list[str]:
        lines: list[str] = []
        anchor = section.section_name.lower().replace(" ", "-").replace("&", "")
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
        badge = self._status_badge(section.verification_status)
        lines.append(f"## {section.section_name} {badge}")
        lines.append("")

        if section.unresolved_issues:
            lines.append(
                f"> ⚠ **{len(section.unresolved_issues)} unresolved issue(s)** remain "
                f"in this section after the maximum revision iterations."
            )
            lines.append("")

        for para in section.paragraphs:
            lines.extend(self._render_paragraph(para))

        lines.append("")
        return lines

    def _render_table(self, table: ReportTable) -> list[str]:
        lines: list[str] = [f"## {table.title}", ""]
        if table.description:
            lines.append(f"*{table.description}*")
            lines.append("")
        if table.columns:
            lines.append("| " + " | ".join(table.columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
            for row in table.rows:
                padded = list(row) + [""] * max(0, len(table.columns) - len(row))
                lines.append("| " + " | ".join(padded[: len(table.columns)]) + " |")
        lines.append("")
        return lines

    def _render_paragraph(self, para: AnalysisParagraph) -> list[str]:
        lines: list[str] = []
        status_note = ""
        if para.verification_status == VerificationStatus.UNRESOLVED:
            status_note = " *(⚠ unresolved – low confidence)*"
        elif para.verification_status == VerificationStatus.PASS:
            status_note = " *(✓ verified)*"

        lines.append(para.text + status_note)
        lines.append("")

        # Inline citation footer
        if para.chunk_ids or para.calc_ids:
            cites: list[str] = []
            for cid in para.chunk_ids:
                cites.append(f"[evidence: `{cid[:8]}…`](#chunk-{cid[:8]})")
            for cid in para.calc_ids:
                cites.append(f"[calc: `{cid[:8]}…`](#calc-{cid[:8]})")
            lines.append(f"<small>Sources: {' | '.join(cites)}</small>")
            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # Appendix rendering
    # ------------------------------------------------------------------

    def _render_evidence_chunk(self, chunk: EvidenceChunk) -> list[str]:
        lines = [
            f"### <a name=\"chunk-{chunk.chunk_id[:8]}\"></a>"
            f"`{chunk.chunk_id[:8]}…` — {chunk.source_type.value}",
            f"**Period:** {chunk.period or 'N/A'}  |  "
            f"**Field:** {chunk.field_name or 'N/A'}",
            "",
            "```",
            chunk.text[:600] + ("…" if len(chunk.text) > 600 else ""),
            "```",
            "",
        ]
        return lines

    def _render_calc(self, calc: CalculationResult) -> list[str]:
        result_str = (
            f"{calc.result:.4f}" if isinstance(calc.result, float) else str(calc.result)[:300]
        )
        lines = [
            f"### <a name=\"calc-{calc.calc_id[:8]}\"></a>"
            f"`{calc.calc_id[:8]}…` — {calc.operation}",
            f"**Description:** {calc.description}",
            f"**Formula:** {calc.formula_description}",
            f"**Result:** `{result_str}`",
        ]
        if calc.error:
            lines.append(f"**Error:** {calc.error}")
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _status_badge(status: VerificationStatus) -> str:
        badges = {
            VerificationStatus.PASS: "✅",
            VerificationStatus.FAIL: "❌",
            VerificationStatus.UNRESOLVED: "⚠️",
            VerificationStatus.SKIPPED: "⏭",
        }
        return badges.get(status, "")
