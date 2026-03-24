"""
ERRGen  —  LangGraph StateGraph

Defines the end-to-end equity research workflow as a typed state machine.
Every stage of the pipeline is a node; shared state flows through edges.

Graph topology
--------------

    START
      │
      ▼
  parse_query          natural-language query → ticker / company / focus / date
      │
      ▼
  retrieve_data        FMP + NewsAPI + Finnhub → EvidenceChunks
      │
      ▼
  extract_evidence     FinancialExtractor (deterministic) + NewsExtractor (LLM)
                       → facts + CalculationResults
      │
      ▼
  analysis_agent       5 section agents → initial paragraph drafts (all FAIL)
      │
      ▼
  verification_agent   CheckerAgent → update paragraph statuses
      │
      ├─── any FAIL? ──► revise_sections   ReviserAgent → revised paragraphs
      │                        │
      │                        └──────────── (loop back to verification_agent)
      │
      └─── all PASS / UNRESOLVED?
                │
                ▼
          prediction_agent   Investment recommendation (gated on upstream quality)
                │
                ▼
          report_writer      Assemble + render + save run artifacts
                │
                ▼
              END

Usage
-----
    from errgen.graph import run

    final_state = run("Write an equity research report for NVIDIA focusing on AI chips.")
    report_md   = final_state["report_md"]
    report_obj  = final_state["report"]

Or with more control:

    from errgen.graph import get_graph, initial_state

    graph = get_graph()
    state = graph.invoke(initial_state("Write a report for AAPL as of 2025-01."))
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from errgen.config import Config
from errgen.nodes import (
    analysis_agent,
    baseline_prediction_agent,
    extract_evidence,
    parse_query,
    prediction_agent,
    report_writer,
    retrieve_data,
    revise_sections,
    verification_agent,
)
from errgen.nodes.verification_agent import route_after_verification
from errgen.state import ErrGenState


def build_graph(mode: str = "full") -> StateGraph:
    """Construct and compile the ERRGen StateGraph."""
    if mode not in {"full", "baseline"}:
        raise ValueError(f"Unsupported graph mode: {mode}")

    graph = StateGraph(ErrGenState)

    # ------------------------------------------------------------------ #
    # Register nodes                                                       #
    # ------------------------------------------------------------------ #
    graph.add_node("parse_query",        parse_query)
    graph.add_node("retrieve_data",      retrieve_data)
    graph.add_node("extract_evidence",   extract_evidence)
    graph.add_node("analysis_agent",     analysis_agent)
    if mode == "full":
        graph.add_node("verification_agent", verification_agent)
        graph.add_node("revise_sections",    revise_sections)
        graph.add_node("prediction_agent",   prediction_agent)
    else:
        graph.add_node("baseline_prediction_agent", baseline_prediction_agent)
    graph.add_node("report_writer",      report_writer)

    # ------------------------------------------------------------------ #
    # Linear edges (deterministic flow)                                    #
    # ------------------------------------------------------------------ #
    graph.add_edge(START,                "parse_query")
    graph.add_edge("parse_query",        "retrieve_data")
    graph.add_edge("retrieve_data",      "extract_evidence")
    graph.add_edge("extract_evidence",   "analysis_agent")
    if mode == "full":
        graph.add_edge("analysis_agent",     "verification_agent")
        graph.add_edge("revise_sections",    "verification_agent")   # loop back
        graph.add_edge("prediction_agent",   "report_writer")
    else:
        graph.add_edge("analysis_agent",     "baseline_prediction_agent")
        graph.add_edge("baseline_prediction_agent", "report_writer")
    graph.add_edge("report_writer",      END)

    # ------------------------------------------------------------------ #
    # Conditional edge: verification loop                                  #
    # ------------------------------------------------------------------ #
    if mode == "full":
        graph.add_conditional_edges(
            "verification_agent",
            route_after_verification,
            {
                "revise_sections":  "revise_sections",
                "prediction_agent": "prediction_agent",
            },
        )

    return graph.compile()


def initial_state(query: str) -> dict:
    """
    Return a minimal initial state dict for graph.invoke().

    Only ``query`` is required; all other fields are populated by nodes.
    This helper ensures every required TypedDict key is present with a
    sensible default so LangGraph does not raise on missing keys.
    """
    return {
        "query":               query,
        "ticker":              "",
        "company_name":        "",
        "focus":               [],
        "as_of_date":          None,
        "run_id":              "",
        "sources":             [],
        "evidence_chunks":     [],
        "extracted_facts":     [],
        "calculation_requests": [],
        "calculations":        [],
        "analysis_sections":   [],
        "paragraph_drafts":    [],
        "checker_verdicts":    [],
        "revision_records":    [],
        "revision_count":      0,
        "has_blocking_issues": False,
        "warnings":            [],
        "errors":              [],
        "report":              None,
        "report_md":           None,
    }


# ---------------------------------------------------------------------------
# Module-level compiled graph (lazy singleton)
# ---------------------------------------------------------------------------

_graphs: dict[str, object] = {}


def get_graph(mode: str = "full"):
    """Return the compiled graph for the requested mode."""
    if mode not in _graphs:
        Config.validate_required()
        _graphs[mode] = build_graph(mode=mode)
    return _graphs[mode]


def run(query: str, mode: str = "full") -> dict:
    """
    Run the full ERRGen pipeline with a natural-language query.

    Args:
        query: e.g. "Write an equity research report for NVIDIA focusing on AI chips."

    Returns:
        Final state dict.  Key fields:
          state["report"]    – FinalReport Pydantic object
          state["report_md"] – rendered Markdown string
          state["run_id"]    – UUID of this run (for locating saved artifacts)
    """
    graph = get_graph(mode=mode)
    return graph.invoke(initial_state(query))
