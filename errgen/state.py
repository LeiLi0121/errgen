"""
LangGraph shared state for ERRGen.

ErrGenState is the single source of truth threaded through every node in
the StateGraph.  Fields annotated with ``Annotated[list, add]`` use
LangGraph's list-append reducer: nodes return only the *new* items and
they are automatically merged into the accumulated list.  Plain fields
are overwritten by whatever the node returns.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Optional, TypedDict

from errgen.models import (
    AnalysisParagraph,
    CalculationRequest,
    CalculationResult,
    CheckerVerdict,
    EvidenceChunk,
    ExtractedFact,
    FinalReport,
    ReportSection,
    RevisionRecord,
    SourceMetadata,
)


class ErrGenState(TypedDict):
    # ------------------------------------------------------------------ #
    # Input                                                                #
    # ------------------------------------------------------------------ #
    query: str                        # raw natural-language request

    # ------------------------------------------------------------------ #
    # Parsed fields  (set once by parse_query, then read-only)            #
    # ------------------------------------------------------------------ #
    ticker: str
    company_name: str
    focus: list[str]                  # inferred focus topics
    as_of_date: Optional[str]         # "YYYY-MM" or None

    # Run identifier written by parse_query
    run_id: str

    # ------------------------------------------------------------------ #
    # Data layer  (append-only – new items returned by each node)         #
    # ------------------------------------------------------------------ #
    sources: Annotated[list[SourceMetadata], add]
    evidence_chunks: Annotated[list[EvidenceChunk], add]

    # ------------------------------------------------------------------ #
    # Extraction layer  (append-only)                                      #
    # ------------------------------------------------------------------ #
    extracted_facts: Annotated[list[ExtractedFact], add]
    calculation_requests: Annotated[list[CalculationRequest], add]
    calculations: Annotated[list[CalculationResult], add]

    # ------------------------------------------------------------------ #
    # Analysis sections  (overwrite – replaced each verification cycle)   #
    # ------------------------------------------------------------------ #
    # analysis_agent writes the initial list; verification_agent and
    # revise_sections replace it with updated paragraph statuses.
    analysis_sections: list[ReportSection]

    # ------------------------------------------------------------------ #
    # Audit logs  (append-only)                                            #
    # ------------------------------------------------------------------ #
    paragraph_drafts: Annotated[list[AnalysisParagraph], add]
    checker_verdicts: Annotated[list[CheckerVerdict], add]
    revision_records: Annotated[list[RevisionRecord], add]

    # ------------------------------------------------------------------ #
    # Verification loop counters                                           #
    # ------------------------------------------------------------------ #
    revision_count: int               # incremented by revise_sections
    has_blocking_issues: bool         # set by prediction_agent gate

    # ------------------------------------------------------------------ #
    # Diagnostics  (append-only)                                           #
    # ------------------------------------------------------------------ #
    warnings: Annotated[list[str], add]
    errors: Annotated[list[str], add]

    # ------------------------------------------------------------------ #
    # Final output  (overwrite)                                            #
    # ------------------------------------------------------------------ #
    report: Optional[FinalReport]
    report_md: Optional[str]
