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
    ReportSection,
    UserRequest,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


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
        for i, section in enumerate(report.sections, 1):
            anchor = section.section_name.lower().replace(" ", "-").replace("&", "")
            anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
            status_badge = self._status_badge(section.verification_status)
            lines.append(f"{i}. [{section.section_name}](#{anchor}) {status_badge}")
        lines.append(f"{len(report.sections) + 1}. [Evidence Appendix](#evidence-appendix)")
        lines.append(f"{len(report.sections) + 2}. [Calculation Appendix](#calculation-appendix)")
        lines.append("")

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
