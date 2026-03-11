"""
ERRGen – Evidence-Grounded Equity Research Report Generator
LangGraph edition

Usage examples
--------------

# Natural-language input (recommended):
python scripts/run_report.py "Write an equity research report for NVIDIA focusing on AI chips."

# With optional overrides:
python scripts/run_report.py \
    "Write a detailed equity research report for Apple as of January 2025, \
     focusing on iPhone revenue, services growth, and AI strategy." \
    --runs-dir /tmp/runs \
    --max-iterations 2 \
    --print-report

# Legacy structured input (still supported):
python scripts/run_report.py --ticker NVDA --as-of 2025-01 \
    --focus "AI chips" "datacenter" "financial analysis"

Prerequisites
-------------
Copy .env.example to .env and fill in:
  OPENAI_API_KEY
  FMP_API_KEY
  NEWSAPI_KEY
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from errgen.config import Config
from errgen.graph import get_graph, initial_state
from errgen.report import ReportRenderer
from errgen.run_record import RunRecord


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ERRGen – Evidence-Grounded Equity Research Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Primary: natural-language query (positional, optional)
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural-language report request, e.g. "
             "\"Write an equity research report for NVIDIA focusing on AI chips.\"",
    )

    # Legacy structured input (still supported for backwards compatibility)
    parser.add_argument(
        "--ticker", default=None,
        help="Stock ticker (e.g. NVDA). Used when --query is not provided.",
    )
    parser.add_argument(
        "--as-of", default=None, metavar="YYYY-MM",
        help="As-of date for the report (e.g. 2025-01).",
    )
    parser.add_argument(
        "--company-name", default=None,
        help="Full company name (optional; inferred from FMP profile if omitted).",
    )
    parser.add_argument(
        "--focus", nargs="+", default=[], metavar="AREA",
        help="Focus areas, e.g. \"AI chips\" \"financial analysis\" \"risks\".",
    )

    # Common options
    parser.add_argument(
        "--runs-dir", default=None,
        help="Directory to save run artifacts (default: ./runs).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override MAX_REVISION_ITERATIONS from .env.",
    )
    parser.add_argument(
        "--print-report", action="store_true",
        help="Print the rendered Markdown report to stdout after generation.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args()


def _build_query(args: argparse.Namespace) -> str:
    """Construct the natural-language query string from CLI arguments."""
    if args.query:
        return args.query

    if not args.ticker:
        print(
            "\n[ERROR] Provide either a natural-language query as the first argument\n"
            "        or use --ticker to specify the stock.\n"
            "        Example: python scripts/run_report.py "
            "\"Write a report for NVIDIA.\"\n",
            file=sys.stderr,
        )
        sys.exit(1)

    ticker = args.ticker.upper()
    as_of_str = f" as of {args.as_of}" if args.as_of else ""
    focus_str = (
        ", ".join(args.focus) if args.focus else "financial analysis and risks"
    )
    if args.company_name:
        subject = f"{args.company_name} ({ticker})"
    else:
        subject = ticker

    return (
        f"Write an equity research report for {subject}{as_of_str} "
        f"focusing on {focus_str}."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("errgen.run_report")

    # Apply config overrides
    if args.runs_dir:
        Config.RUNS_DIR = args.runs_dir
    if args.max_iterations is not None:
        Config.MAX_REVISION_ITERATIONS = args.max_iterations

    query = _build_query(args)

    log.info("Starting ERRGen (LangGraph)")
    log.info("  Query:        %s", query[:120])
    log.info("  Runs dir:     %s", Config.RUNS_DIR)
    log.info("  Max iters:    %d", Config.MAX_REVISION_ITERATIONS)

    # ------------------------------------------------------------------
    # Run the LangGraph pipeline
    # ------------------------------------------------------------------
    try:
        graph = get_graph()
        final_state = graph.invoke(initial_state(query))
    except ValueError as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        log.exception("Graph execution failed: %s", exc)
        print(f"\n[ERROR] Pipeline failed: {exc}\n", file=sys.stderr)
        sys.exit(1)

    report = final_state.get("report")
    report_md = final_state.get("report_md", "")
    run_id = final_state.get("run_id", "")

    if not report:
        print("\n[ERROR] No report was produced.\n", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Report generated : {report.report_id}")
    print(f"Ticker           : {report.request.ticker}")
    print(f"Company          : {report.request.company_name or '(inferred)'}")
    print(f"Verification     : {report.overall_status.value.upper()}")
    print(f"Sections         : {len(report.sections)}")
    print(f"Cited chunks     : {len(report.evidence_appendix)}")
    print(f"Calculations     : {len(report.calculation_appendix)}")

    if final_state.get("warnings"):
        print(f"Warnings         : {len(final_state['warnings'])}")
        for w in final_state["warnings"]:
            print(f"  ⚠  {w}")

    # Locate saved run directory
    if run_id:
        run_dir = Path(Config.RUNS_DIR) / run_id
        if run_dir.exists():
            print(f"\nRun artifacts    : {run_dir}")
            report_path = run_dir / "report.md"
            if report_path.exists():
                print(f"Markdown report  : {report_path}")

    print("=" * 70 + "\n")

    # Optionally stream Markdown to stdout
    if args.print_report and report_md:
        print(report_md)


if __name__ == "__main__":
    main()
