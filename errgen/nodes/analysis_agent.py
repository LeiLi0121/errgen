"""
analysis_agent node

Runs all five section-level analysis agents in sequence:

  1. Company Overview       – built directly from profile chunks; no LLM needed
  2. Recent Developments    – NewsAnalysisAgent
  3. Financial Analysis     – FinancialAnalysisAgent
  4. Business & Competitive – BusinessAnalysisAgent
  5. Risk Analysis          – RiskAnalysisAgent

Each agent receives only the evidence subset relevant to its section.
All generated paragraphs start at VerificationStatus.FAIL – the
verification / revision loop (next nodes) will update their status.

Returns:
  analysis_sections: initial list[ReportSection] (overwrites state)
  paragraph_drafts:  all draft paragraphs (appended to audit log)
"""

from __future__ import annotations

import logging

from errgen.analysis.business import BusinessAnalysisAgent
from errgen.analysis.financial import FinancialAnalysisAgent
from errgen.analysis.news import NewsAnalysisAgent
from errgen.analysis.risk import RiskAnalysisAgent
from errgen.models import (
    AnalysisParagraph,
    CalculationResult,
    Citation,
    EvidenceChunk,
    ReportSection,
    SourceType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def _select_chunks(section_key: str, all_chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
    """Route evidence chunks to the relevant analysis agent."""
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
                SourceType.FILING,
            )
        ]

    # risk_analysis receives all chunks
    return all_chunks


def _build_overview_section(
    all_chunks: list[EvidenceChunk],
    section_order: int,
) -> ReportSection:
    """
    Build Company Overview directly from company profile chunks.
    No LLM generation required – profile data is treated as factual.
    """
    profile_chunks = [c for c in all_chunks if c.source_type == SourceType.COMPANY_PROFILE]
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
        paragraphs.append(
            AnalysisParagraph(
                section_name="Company Overview",
                text=chunk.text,
                citations=citations,
                chunk_ids=[chunk.chunk_id],
                calc_ids=[],
                verification_status=VerificationStatus.PASS,  # factual profile data
            )
        )

    return ReportSection(
        section_name="Company Overview",
        section_order=section_order,
        paragraphs=paragraphs,
        verification_status=VerificationStatus.PASS,
    )


def analysis_agent(state: dict) -> dict:
    """Generate initial paragraph drafts for all five analysis sections."""
    ticker: str = state["ticker"]
    as_of: str | None = state.get("as_of_date")
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    focus: list[str] = state.get("focus") or []
    query: str = state.get("query", "")

    request_context = query
    if focus:
        request_context += f"\nFocus areas: {', '.join(focus)}"

    logger.info("analysis_agent: generating sections for %s", ticker)

    agent_configs = [
        ("recent_developments", NewsAnalysisAgent()),
        ("financial_analysis",  FinancialAnalysisAgent()),
        ("business_analysis",   BusinessAnalysisAgent()),
        ("risk_analysis",       RiskAnalysisAgent()),
    ]

    sections: list[ReportSection] = []
    all_drafts: list[AnalysisParagraph] = []

    # 1. Company Overview (no LLM)
    overview = _build_overview_section(all_chunks, section_order=1)
    sections.append(overview)

    # 2–5. LLM-based sections
    for order, (key, agent) in enumerate(agent_configs, start=2):
        relevant = _select_chunks(key, all_chunks)
        logger.info(
            "analysis_agent: '%s' → %d relevant chunks",
            agent.section_name, len(relevant),
        )
        paragraphs = agent.generate(
            ticker=ticker,
            request_context=request_context,
            chunks=relevant,
            calc_results=calc_results,
            as_of_date=as_of,
        )
        all_drafts.extend(paragraphs)
        sections.append(
            ReportSection(
                section_name=agent.section_name,
                section_order=order,
                paragraphs=paragraphs,
                verification_status=VerificationStatus.FAIL,
            )
        )

    logger.info(
        "analysis_agent: %d sections, %d draft paragraphs total",
        len(sections), len(all_drafts),
    )
    return {
        "analysis_sections": sections,
        "paragraph_drafts": all_drafts,
    }
