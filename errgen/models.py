"""
Core data models for the ERRGen (Equity Research Report Generator) system.

Every object in the pipeline is represented here as a Pydantic v2 model,
ensuring type safety, serialisability, and easy persistence to JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    NEWS = "news"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    COMPANY_PROFILE = "company_profile"
    FILING = "filing"
    PRICE_DATA = "price_data"
    PEER_DATA = "peer_data"
    OTHER = "other"


class IssueType(str, Enum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_MISMATCH = "citation_mismatch"
    HALLUCINATION = "hallucination"
    NUMERICAL_ERROR = "numerical_error"
    INTERNAL_INCONSISTENCY = "internal_inconsistency"
    SCOPE_VIOLATION = "scope_violation"
    OVERCLAIMING = "overclaiming"


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNRESOLVED = "unresolved"  # max iterations reached without passing
    SKIPPED = "skipped"        # checker was not run (e.g. no evidence)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class UserRequest(BaseModel):
    """Parsed user request for an equity research report."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str
    ticker: str
    company_name: Optional[str] = None
    market: str = "US"
    as_of_date: Optional[str] = None   # e.g. "2025-01" or "2025-01-31"
    focus_areas: list[str] = Field(default_factory=list)
    report_style: str = "standard"
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Evidence layer
# ---------------------------------------------------------------------------


class SourceMetadata(BaseModel):
    """Metadata for a retrieved data source (API call, filing, article)."""

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType
    url: Optional[str] = None
    api_source: str                       # e.g. "fmp", "newsapi"
    document_identifier: Optional[str] = None   # filing date, article id, etc.
    filing_section: Optional[str] = None
    statement_field: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    ticker: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    """
    A discrete piece of evidence that analysis agents can cite.

    Chunks come from raw API data that has been segmented and labelled.
    The chunk_id is the stable citation key used throughout the pipeline.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    source_type: SourceType
    text: str                             # human-readable text of this chunk
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Financial-specific fields (populated for statement chunks)
    period: Optional[str] = None          # e.g. "FY2024", "Q3 2024"
    field_name: Optional[str] = None      # e.g. "revenue", "netIncome"
    numeric_value: Optional[float] = None
    unit: Optional[str] = None            # e.g. "USD", "USD_millions", "ratio"


class ExtractedFact(BaseModel):
    """
    A structured fact derived from one or more evidence chunks,
    suitable for calculator input or direct citation.
    """

    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_ids: list[str]
    fact_type: str          # e.g. "revenue", "net_income", "news_event"
    subject: str            # ticker or company name
    period: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    description: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class CalculationRequest(BaseModel):
    """A deterministic calculation requested by the pipeline."""

    calc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: str          # e.g. "growth_rate", "margin", "ratio"
    inputs: dict[str, Any]
    description: str        # human-readable intent
    chunk_ids_used: list[str] = Field(default_factory=list)


class CalculationResult(BaseModel):
    """The output of a deterministic calculation."""

    calc_id: str
    operation: str
    inputs: dict[str, Any]
    result: Union[float, Dict[str, Any]]
    formula_description: str
    description: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Analysis output
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A paragraph-level citation linking a claim to an evidence chunk."""

    citation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    source_id: str
    source_type: SourceType
    text_snippet: str         # the exact chunk text passage being cited
    relevance_note: Optional[str] = None


class AnalysisParagraph(BaseModel):
    """
    A single analysis paragraph in a report section.

    Carries explicit references to the evidence chunks and calculation
    results that support it. Verification status is updated in place as
    the checker/reviser loop runs.
    """

    paragraph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_name: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)    # fast lookup
    calc_ids: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.FAIL
    checker_verdicts: list[CheckerVerdict] = Field(default_factory=list)
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    iteration: int = 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class CheckerIssue(BaseModel):
    """A single issue found by the checker agent."""

    issue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issue_type: IssueType
    severity: IssueSeverity
    paragraph_id: str
    offending_span: Optional[str] = None   # the problematic substring
    explanation: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_calc_ids: list[str] = Field(default_factory=list)
    recommended_fix: str


class CheckerVerdict(BaseModel):
    """The overall verdict of a checker run on a single paragraph."""

    verdict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paragraph_id: str
    status: VerificationStatus
    issues: list[CheckerIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    iteration: int = 0


class RevisionRecord(BaseModel):
    """Records a single revision attempt for a paragraph."""

    revision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paragraph_id: str
    iteration: int
    original_text: str
    revised_text: str
    issues_addressed: list[str] = Field(default_factory=list)  # issue_ids
    changes_summary: str = ""
    revised_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """One section of the final report (e.g. 'Financial Analysis')."""

    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_name: str
    section_order: int
    paragraphs: list[AnalysisParagraph] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.FAIL
    unresolved_issues: list[CheckerIssue] = Field(default_factory=list)


class ReportTable(BaseModel):
    """Structured deterministic table rendered in the final report."""

    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    description: str = ""


class FinalReport(BaseModel):
    """
    The assembled equity research report.

    Contains all sections, the evidence appendix (all cited chunks),
    and a calculation appendix (all computation traces).
    """

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: UserRequest
    sections: list[ReportSection] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    evidence_appendix: list[EvidenceChunk] = Field(default_factory=list)
    calculation_appendix: list[CalculationResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    overall_status: VerificationStatus = VerificationStatus.FAIL
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run artifact (full experiment record)
# ---------------------------------------------------------------------------


class RunArtifact(BaseModel):
    """
    Complete record of a single pipeline run.

    Everything needed to reconstruct what happened, debug failures, and
    trace any sentence in the final report back to its original evidence.
    """

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request: UserRequest
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "running"   # running | completed | failed

    # Data collection
    retrieved_sources: list[SourceMetadata] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)

    # Extraction
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)

    # Calculations
    calculation_requests: list[CalculationRequest] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)

    # Analysis (drafts before assembly)
    paragraph_drafts: list[AnalysisParagraph] = Field(default_factory=list)

    # Verification
    checker_verdicts: list[CheckerVerdict] = Field(default_factory=list)
    revision_records: list[RevisionRecord] = Field(default_factory=list)

    # Final output
    final_report: Optional[FinalReport] = None

    # Diagnostics
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def chunk_by_id(self, chunk_id: str) -> Optional[EvidenceChunk]:
        for c in self.evidence_chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def calc_by_id(self, calc_id: str) -> Optional[CalculationResult]:
        for c in self.calculation_results:
            if c.calc_id == calc_id:
                return c
        return None


# ---------------------------------------------------------------------------
# Forward-reference resolution
# ---------------------------------------------------------------------------
# Pydantic v2 requires explicit model_rebuild() when models reference
# each other before the referenced class is defined.

AnalysisParagraph.model_rebuild()
