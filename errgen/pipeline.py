"""
Main pipeline orchestrator.

Implements the full end-to-end ERRGen flow:

  A. Request parsing (input → UserRequest)
  B. Data collection (FMP + NewsAPI → EvidenceChunk list)
  C. Information extraction (chunks → ExtractedFact + CalculationResult)
  D. Section-level analysis  (facts + chunks + calcs → paragraph drafts)
  E. Verification loop       (checker → reviser → re-checker, per paragraph)
  F. Prediction generation   (ONLY after upstream sections pass)
  G. Report assembly         (sections → FinalReport)
  H. Run artifact persistence

IMPORTANT: Prediction generation is gated behind the verification stage.
The pipeline will NOT call the prediction agent if any upstream section
has critical unresolved issues.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from errgen.analysis.base import BaseAnalysisAgent, _format_chunks_for_prompt, _format_calcs_for_prompt
from errgen.analysis.business import BusinessAnalysisAgent
from errgen.analysis.financial import FinancialAnalysisAgent
from errgen.analysis.news import NewsAnalysisAgent
from errgen.analysis.risk import RiskAnalysisAgent
from errgen.calculator import FinanceCalculator
from errgen.config import Config
from errgen.data import FMPClient, NewsAPIClient
from errgen.extraction import FinancialExtractor, NewsExtractor
from errgen.llm import build_messages, chat_json
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    EvidenceChunk,
    FinalReport,
    IssueSeverity,
    ReportSection,
    RunArtifact,
    SourceMetadata,
    SourceType,
    UserRequest,
    VerificationStatus,
)
from errgen.report import ReportAssembler
from errgen.run_record import RunRecord
from errgen.verification import CheckerAgent, ReviserAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction agent (inline – gated, runs last)
# ---------------------------------------------------------------------------

_PREDICTION_SYSTEM = """You are a senior equity research analyst writing the
'Investment Recommendation & Outlook' section of a research report.

This is the final section. It synthesises the verified analysis above into a
forward-looking view.

RULES:
1. Base your recommendation ONLY on the verified section summaries and evidence provided.
2. Cite chunk_ids and calc_ids that support your recommendation.
3. Acknowledge key risks and uncertainties explicitly.
4. Use language appropriate to the evidence certainty level.
5. Do NOT promise returns or guarantee outcomes.
6. State your overall stance: Positive / Neutral / Cautious / Negative (and why).
7. If evidence is insufficient for a firm recommendation, say so clearly.

Return JSON:
{
  "paragraphs": [
    {
      "text": "...",
      "chunk_ids": ["id1"],
      "calc_ids": ["calc1"]
    }
  ]
}
"""


class PredictionAgent(BaseAnalysisAgent):
    """Generates the final investment recommendation section."""

    section_name = "Investment Recommendation & Outlook"

    def _build_system_prompt(self, as_of_date: str | None) -> str:
        return _PREDICTION_SYSTEM

    def _build_user_prompt(
        self,
        ticker: str,
        request_context: str,
        chunk_block: str,
        calc_block: str,
        as_of_date: str | None,
    ) -> str:
        return (
            f"Ticker: {ticker}\n"
            f"Report context: {request_context}\n"
            f"As-of date: {as_of_date or 'not specified'}\n\n"
            f"=== VERIFIED SECTION SUMMARIES ===\n{request_context}\n\n"
            f"=== KEY EVIDENCE CHUNKS ===\n{chunk_block}\n\n"
            f"=== CALCULATION RESULTS ===\n{calc_block}\n\n"
            f"Write the 'Investment Recommendation & Outlook' section."
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    End-to-end equity research report generation pipeline.

    Usage:
        pipeline = Pipeline()
        request = UserRequest(
            raw_text="Write an equity research report for NVIDIA...",
            ticker="NVDA",
            as_of_date="2025-01",
            focus_areas=["AI chips", "financial analysis", "risks"],
        )
        report = pipeline.run(request)
    """

    def __init__(self) -> None:
        Config.validate_required()

        self.fmp = FMPClient()
        self.newsapi = NewsAPIClient()
        self.calculator = FinanceCalculator()
        self.financial_extractor = FinancialExtractor()
        self.news_extractor = NewsExtractor()
        self.checker = CheckerAgent()
        self.reviser = ReviserAgent()
        self.assembler = ReportAssembler()
        self.run_record = RunRecord()

        self._analysis_agents: list[tuple[str, BaseAnalysisAgent]] = [
            ("company_overview", None),     # built from profile chunks directly
            ("recent_developments", NewsAnalysisAgent()),
            ("financial_analysis", FinancialAnalysisAgent()),
            ("business_analysis", BusinessAnalysisAgent()),
            ("risk_analysis", RiskAnalysisAgent()),
        ]
        self._prediction_agent = PredictionAgent()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, request: UserRequest) -> FinalReport:
        """Run the full pipeline and return the final report."""
        artifact = RunArtifact(request=request)
        logger.info(
            "Pipeline starting | run_id=%s | ticker=%s | as_of_date=%s",
            artifact.run_id, request.ticker, request.as_of_date,
        )

        try:
            # B. Data collection
            sources, all_chunks = self._collect_data(request)
            artifact.retrieved_sources = sources
            artifact.evidence_chunks = all_chunks

            # C. Extraction + calculations
            facts, calc_reqs, calc_results = self._extract_and_calculate(
                request, all_chunks
            )
            artifact.extracted_facts = facts
            artifact.calculation_requests = calc_reqs
            artifact.calculation_results = calc_results

            # D + E. Analysis + verification loop (all sections except prediction)
            sections = self._run_analysis_sections(request, all_chunks, calc_results, artifact)

            # F. Prediction (gated)
            prediction_section = self._run_prediction_section(
                request, sections, all_chunks, calc_results, artifact
            )
            if prediction_section:
                sections.append(prediction_section)

            # G. Assemble report
            report = self.assembler.assemble(
                request=request,
                sections=sections,
                all_chunks=all_chunks,
                calc_results=calc_results,
                warnings=artifact.warnings,
            )
            artifact.final_report = report
            artifact.status = "completed"
            artifact.completed_at = datetime.utcnow()

            logger.info(
                "Pipeline completed | run_id=%s | overall_status=%s",
                artifact.run_id, report.overall_status.value,
            )

        except Exception as exc:
            artifact.status = "failed"
            artifact.completed_at = datetime.utcnow()
            artifact.errors.append(str(exc))
            logger.exception("Pipeline failed | run_id=%s | error=%s", artifact.run_id, exc)
            raise

        finally:
            self.run_record.save(artifact)

        return artifact.final_report

    # ------------------------------------------------------------------
    # B. Data collection
    # ------------------------------------------------------------------

    def _collect_data(
        self, request: UserRequest
    ) -> tuple[list[SourceMetadata], list[EvidenceChunk]]:
        ticker = request.ticker
        as_of = request.as_of_date
        company_name = request.company_name

        # Determine date range for news
        from_date, to_date = self._date_range_for_news(as_of)

        sources: list[SourceMetadata] = []
        chunks: list[EvidenceChunk] = []

        # -- Company profile
        logger.info("Collecting company profile for %s...", ticker)
        src, cks = self.fmp.get_company_profile(ticker)
        sources.append(src)
        chunks.extend(cks)

        # Infer company name from profile if not provided
        if not company_name:
            for ck in cks:
                if ck.field_name == "overview":
                    # Extract company name from the overview chunk text
                    # Format: "CompanyName (TICKER) is a ..."
                    try:
                        request.company_name = ck.text.split("(")[0].strip()
                    except Exception:
                        pass
                    break

        # -- Income statement (annual, last 4 years)
        logger.info("Collecting income statement for %s...", ticker)
        src, cks = self.fmp.get_income_statement(ticker, "annual")
        sources.append(src)
        chunks.extend(cks)

        # -- Balance sheet (annual)
        logger.info("Collecting balance sheet for %s...", ticker)
        src, cks = self.fmp.get_balance_sheet(ticker, "annual")
        sources.append(src)
        chunks.extend(cks)

        # -- Cash flow (annual)
        logger.info("Collecting cash flow statement for %s...", ticker)
        src, cks = self.fmp.get_cash_flow(ticker, "annual")
        sources.append(src)
        chunks.extend(cks)

        # -- FMP stock news
        logger.info("Collecting FMP news for %s...", ticker)
        try:
            src, cks = self.fmp.get_stock_news(
                ticker,
                limit=Config.MAX_NEWS_ARTICLES,
                from_date=from_date,
                to_date=to_date,
            )
            sources.append(src)
            chunks.extend(cks)
        except Exception as exc:
            msg = f"FMP news collection failed: {exc}"
            logger.warning(msg)

        # -- NewsAPI (supplemental)
        logger.info("Collecting NewsAPI articles for %s...", ticker)
        try:
            src, cks = self.newsapi.get_company_news(
                ticker=ticker,
                company_name=request.company_name,
                from_date=from_date,
                to_date=to_date,
                page_size=Config.MAX_NEWS_ARTICLES,
            )
            sources.append(src)
            chunks.extend(cks)
        except Exception as exc:
            msg = f"NewsAPI collection failed: {exc}"
            logger.warning(msg)

        logger.info(
            "Data collection complete: %d sources, %d chunks", len(sources), len(chunks)
        )
        return sources, chunks

    # ------------------------------------------------------------------
    # C. Extraction + calculations
    # ------------------------------------------------------------------

    def _extract_and_calculate(
        self,
        request: UserRequest,
        all_chunks: list[EvidenceChunk],
    ):
        ticker = request.ticker
        news_chunks = [c for c in all_chunks if c.source_type == SourceType.NEWS]

        # Financial extraction (deterministic)
        facts, calc_reqs, calc_results = self.financial_extractor.extract(
            ticker, all_chunks
        )

        # News extraction (LLM-based)
        news_facts = self.news_extractor.extract(
            ticker=ticker,
            news_chunks=news_chunks,
            as_of_date=request.as_of_date,
        )
        facts.extend(news_facts)

        return facts, calc_reqs, calc_results

    # ------------------------------------------------------------------
    # D + E. Analysis sections + verification loop
    # ------------------------------------------------------------------

    def _run_analysis_sections(
        self,
        request: UserRequest,
        all_chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        artifact: RunArtifact,
    ) -> list[ReportSection]:
        sections: list[ReportSection] = []
        section_order = 0

        for section_key, agent in self._analysis_agents:
            section_order += 1

            if section_key == "company_overview":
                section = self._build_company_overview_section(
                    request, all_chunks, section_order
                )
                sections.append(section)
                continue

            # Filter relevant chunks per section
            relevant_chunks = self._select_chunks_for_section(
                section_key, all_chunks
            )

            logger.info(
                "Generating section '%s' (%d relevant chunks)...",
                agent.section_name, len(relevant_chunks),
            )

            # Generate initial paragraph drafts
            paragraphs = agent.generate(
                ticker=request.ticker,
                request_context=request.raw_text,
                chunks=relevant_chunks,
                calc_results=calc_results,
                as_of_date=request.as_of_date,
            )

            # Store drafts for audit
            artifact.paragraph_drafts.extend(paragraphs)

            # Run verification loop for each paragraph
            verified_paragraphs = []
            verified_context_parts: list[str] = []

            for para in paragraphs:
                verified_para = self._verify_paragraph_loop(
                    paragraph=para,
                    chunks=relevant_chunks,
                    calc_results=calc_results,
                    as_of_date=request.as_of_date,
                    verified_context="\n".join(verified_context_parts),
                    artifact=artifact,
                )
                verified_paragraphs.append(verified_para)

                # Accumulate verified content for consistency checking
                if verified_para.verification_status == VerificationStatus.PASS:
                    verified_context_parts.append(
                        f"[{agent.section_name}] {verified_para.text[:200]}"
                    )

            # Determine section-level status
            unresolved = [
                issue
                for para in verified_paragraphs
                for v in para.checker_verdicts
                for issue in v.issues
                if issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)
                and para.verification_status == VerificationStatus.UNRESOLVED
            ]

            section_status = (
                VerificationStatus.PASS
                if all(
                    p.verification_status == VerificationStatus.PASS
                    for p in verified_paragraphs
                )
                else (
                    VerificationStatus.UNRESOLVED
                    if any(
                        p.verification_status == VerificationStatus.UNRESOLVED
                        for p in verified_paragraphs
                    )
                    else VerificationStatus.FAIL
                )
            )

            section = ReportSection(
                section_name=agent.section_name,
                section_order=section_order,
                paragraphs=verified_paragraphs,
                verification_status=section_status,
                unresolved_issues=unresolved,
            )
            sections.append(section)

            if unresolved:
                artifact.warnings.append(
                    f"Section '{agent.section_name}' has {len(unresolved)} "
                    f"unresolved blocking issues after max iterations."
                )
                logger.warning(
                    "Section '%s' has %d unresolved issues.",
                    agent.section_name, len(unresolved),
                )

        return sections

    def _verify_paragraph_loop(
        self,
        paragraph: AnalysisParagraph,
        chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        as_of_date: str | None,
        verified_context: str,
        artifact: RunArtifact,
    ) -> AnalysisParagraph:
        """
        Run the generate → check → revise → re-check loop for one paragraph.

        Stops when:
          - Checker returns PASS, OR
          - max iterations reached (marks paragraph as UNRESOLVED)
        """
        current_para = paragraph
        max_iter = Config.MAX_REVISION_ITERATIONS

        for iteration in range(max_iter + 1):
            verdict = self.checker.check(
                paragraph=current_para,
                chunks=chunks,
                calc_results=calc_results,
                iteration=iteration,
                as_of_date=as_of_date,
                verified_context=verified_context,
            )

            current_para.checker_verdicts.append(verdict)
            artifact.checker_verdicts.append(verdict)

            if verdict.status == VerificationStatus.PASS:
                current_para.verification_status = VerificationStatus.PASS
                logger.info(
                    "Paragraph %s PASSED on iteration %d",
                    current_para.paragraph_id, iteration,
                )
                break

            if iteration >= max_iter:
                # Max iterations reached – mark as unresolved
                current_para.verification_status = VerificationStatus.UNRESOLVED
                logger.warning(
                    "Paragraph %s UNRESOLVED after %d iterations",
                    current_para.paragraph_id, max_iter,
                )
                break

            # Revise the paragraph
            revised_para, revision_record = self.reviser.revise(
                paragraph=current_para,
                verdict=verdict,
                chunks=chunks,
                calc_results=calc_results,
                iteration=iteration + 1,
            )
            current_para = revised_para
            current_para.revision_history.append(revision_record)
            artifact.revision_records.append(revision_record)

        return current_para

    # ------------------------------------------------------------------
    # F. Prediction (gated)
    # ------------------------------------------------------------------

    def _run_prediction_section(
        self,
        request: UserRequest,
        sections: list[ReportSection],
        all_chunks: list[EvidenceChunk],
        calc_results: list[CalculationResult],
        artifact: RunArtifact,
    ) -> ReportSection | None:
        """
        Generate the prediction / recommendation section.

        GATE: If any upstream section has critical unresolved issues, skip
        prediction and emit a warning rather than generate a potentially
        unreliable recommendation.
        """
        # Check for blocking unresolved issues
        blocking_sections = [
            s.section_name
            for s in sections
            if s.unresolved_issues
        ]
        if blocking_sections:
            msg = (
                f"Prediction section SKIPPED because the following sections have "
                f"unresolved critical/major issues: {blocking_sections}. "
                f"Fix those sections before generating a recommendation."
            )
            artifact.warnings.append(msg)
            logger.warning(msg)
            return None

        # Build verified section summary as context for the prediction agent
        section_summary = self._build_section_summary(sections)

        # Key evidence: financial, news, and calculations
        key_chunks = (
            [c for c in all_chunks if c.source_type != SourceType.NEWS][:20]
            + [c for c in all_chunks if c.source_type == SourceType.NEWS][:10]
        )

        logger.info("Generating prediction section...")

        paragraphs = self._prediction_agent.generate(
            ticker=request.ticker,
            request_context=section_summary,  # pass verified summaries as context
            chunks=key_chunks,
            calc_results=calc_results,
            as_of_date=request.as_of_date,
        )

        artifact.paragraph_drafts.extend(paragraphs)

        # Also verify prediction paragraphs
        verified_paragraphs = []
        for para in paragraphs:
            verified_para = self._verify_paragraph_loop(
                paragraph=para,
                chunks=key_chunks,
                calc_results=calc_results,
                as_of_date=request.as_of_date,
                verified_context=section_summary,
                artifact=artifact,
            )
            verified_paragraphs.append(verified_para)

        unresolved = [
            issue
            for para in verified_paragraphs
            for v in para.checker_verdicts
            for issue in v.issues
            if para.verification_status == VerificationStatus.UNRESOLVED
        ]

        section_status = (
            VerificationStatus.PASS
            if all(p.verification_status == VerificationStatus.PASS for p in verified_paragraphs)
            else VerificationStatus.UNRESOLVED
        )

        return ReportSection(
            section_name="Investment Recommendation & Outlook",
            section_order=len(sections) + 1,
            paragraphs=verified_paragraphs,
            verification_status=section_status,
            unresolved_issues=unresolved,
        )

    # ------------------------------------------------------------------
    # Company overview section (direct from profile chunks, no LLM)
    # ------------------------------------------------------------------

    def _build_company_overview_section(
        self,
        request: UserRequest,
        all_chunks: list[EvidenceChunk],
        section_order: int,
    ) -> ReportSection:
        """
        Build the Company Overview section directly from company profile chunks.
        No LLM generation needed – the profile data speaks for itself.
        """
        from errgen.models import Citation, VerificationStatus

        profile_chunks = [
            c for c in all_chunks if c.source_type == SourceType.COMPANY_PROFILE
        ]

        paragraphs: list[AnalysisParagraph] = []
        for chunk in profile_chunks:
            citations = [
                Citation(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    text_snippet=chunk.text[:300],
                )
            ]
            para = AnalysisParagraph(
                section_name="Company Overview",
                text=chunk.text,
                citations=citations,
                chunk_ids=[chunk.chunk_id],
                calc_ids=[],
                verification_status=VerificationStatus.PASS,  # profile data = factual
            )
            paragraphs.append(para)

        return ReportSection(
            section_name="Company Overview",
            section_order=section_order,
            paragraphs=paragraphs,
            verification_status=VerificationStatus.PASS,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_chunks_for_section(
        section_key: str,
        all_chunks: list[EvidenceChunk],
    ) -> list[EvidenceChunk]:
        """Route chunks to the relevant analysis agent."""
        if section_key == "recent_developments":
            return [c for c in all_chunks if c.source_type == SourceType.NEWS]

        if section_key == "financial_analysis":
            return [
                c for c in all_chunks
                if c.source_type in (
                    SourceType.INCOME_STATEMENT,
                    SourceType.BALANCE_SHEET,
                    SourceType.CASH_FLOW,
                )
            ]

        if section_key == "business_analysis":
            return [
                c for c in all_chunks
                if c.source_type in (
                    SourceType.COMPANY_PROFILE,
                    SourceType.NEWS,
                )
            ]

        if section_key == "risk_analysis":
            # Risk uses all chunk types
            return all_chunks

        return all_chunks

    @staticmethod
    def _date_range_for_news(as_of_date: str | None) -> tuple[str | None, str | None]:
        """Return (from_date, to_date) suitable for news API queries."""
        if not as_of_date:
            return None, None
        # Simple heuristic: pull 12 months of news up to as-of date
        try:
            if len(as_of_date) == 7:  # "YYYY-MM"
                year, month = int(as_of_date[:4]), int(as_of_date[5:7])
            else:
                parts = as_of_date.split("-")
                year, month = int(parts[0]), int(parts[1])
            from_year = year - 1
            from_date = f"{from_year}-{month:02d}-01"
            to_date = as_of_date if len(as_of_date) > 7 else f"{as_of_date}-28"
            return from_date, to_date
        except Exception:
            return None, None

    @staticmethod
    def _build_section_summary(sections: list[ReportSection]) -> str:
        """Summarise verified sections into a compact context string."""
        lines: list[str] = []
        for section in sections:
            lines.append(f"## {section.section_name} (status: {section.verification_status.value})")
            for para in section.paragraphs:
                if para.verification_status == VerificationStatus.PASS:
                    lines.append(para.text[:300] + ("..." if len(para.text) > 300 else ""))
            lines.append("")
        return "\n".join(lines)
