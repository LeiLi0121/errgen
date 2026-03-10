"""
Entry point script for running an equity research report via command line.

Usage examples
--------------
# Minimal – NVIDIA as of January 2025
python scripts/run_report.py --ticker NVDA --as-of 2025-01

# With focus areas
python scripts/run_report.py \
    --ticker NVDA \
    --as-of 2025-01 \
    --focus "AI chips" "financial analysis" "datacenter" "risks" \
    --request "Write an equity research report for NVIDIA as of Jan 2025 \
               focusing on AI chips, data center growth, financial analysis, \
               and investment risks."

# Custom output directory
python scripts/run_report.py --ticker AAPL --as-of 2025-01 --runs-dir /tmp/runs

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

# Ensure project root is on the path when running as a script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from errgen.config import Config
from errgen.models import UserRequest
from errgen.pipeline import Pipeline
from errgen.report import ReportRenderer
from errgen.run_record import RunRecord


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy HTTP request logs unless verbose
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
    )
    parser.add_argument(
        "--ticker",
        required=True,
        type=str,
        help="Stock ticker symbol (e.g. NVDA, AAPL, MSFT)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM",
        help="As-of date for the report (e.g. 2025-01). "
             "News and data are filtered to this date.",
    )
    parser.add_argument(
        "--company-name",
        default=None,
        help="Optional: full company name (used for news search). "
             "If omitted, fetched from FMP company profile.",
    )
    parser.add_argument(
        "--focus",
        nargs="+",
        default=[],
        metavar="AREA",
        help="Focus areas for the report (e.g. 'AI chips' 'financial analysis' 'risks')",
    )
    parser.add_argument(
        "--request",
        default=None,
        help="Full natural-language request text. "
             "If omitted, a standard request is generated from ticker + focus areas.",
    )
    parser.add_argument(
        "--market",
        default="US",
        help="Market (default: US)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory to save run artifacts (default: ./runs)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override MAX_REVISION_ITERATIONS from .env",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the rendered Markdown report to stdout after generation.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("errgen.run_report")

    # Apply overrides
    if args.runs_dir:
        Config.RUNS_DIR = args.runs_dir
    if args.max_iterations is not None:
        Config.MAX_REVISION_ITERATIONS = args.max_iterations

    # Build raw request text
    if args.request:
        raw_text = args.request
    else:
        focus_str = ", ".join(args.focus) if args.focus else "financial analysis and risks"
        as_of_str = f" as of {args.as_of}" if args.as_of else ""
        raw_text = (
            f"Write an equity research report for {args.ticker.upper()}{as_of_str} "
            f"focusing on {focus_str}."
        )

    request = UserRequest(
        raw_text=raw_text,
        ticker=args.ticker.upper(),
        company_name=args.company_name,
        market=args.market,
        as_of_date=args.as_of,
        focus_areas=args.focus,
    )

    log.info("Starting ERRGen pipeline")
    log.info("  Ticker:       %s", request.ticker)
    log.info("  As-of date:   %s", request.as_of_date or "(not set)")
    log.info("  Focus areas:  %s", request.focus_areas or "(default)")
    log.info("  Runs dir:     %s", Config.RUNS_DIR)
    log.info("  Max iters:    %d", Config.MAX_REVISION_ITERATIONS)

    # Run pipeline
    try:
        pipeline = Pipeline()
        report = pipeline.run(request)
    except ValueError as exc:
        # Configuration / API key errors
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        print(f"\n[ERROR] Pipeline failed: {exc}\n", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 70)
    print(f"Report generated: {report.report_id}")
    print(f"Ticker:          {report.request.ticker}")
    print(f"Status:          {report.overall_status.value.upper()}")
    print(f"Sections:        {len(report.sections)}")
    print(f"Cited chunks:    {len(report.evidence_appendix)}")
    print(f"Calculations:    {len(report.calculation_appendix)}")
    if report.warnings:
        print(f"Warnings:        {len(report.warnings)}")
        for w in report.warnings:
            print(f"  ⚠  {w}")

    # Locate saved run
    rr = RunRecord()
    run_id = None
    for run in rr.list_runs():
        if run.get("overall_report_status") is not None:
            run_id = run.get("run_id")
            break

    if run_id:
        run_dir = Path(Config.RUNS_DIR) / run_id
        print(f"\nRun artifacts:   {run_dir}")
        report_path = run_dir / "report.md"
        if report_path.exists():
            print(f"Markdown report: {report_path}")

    print("=" * 70 + "\n")

    # Optionally print to stdout
    if args.print_report:
        renderer = ReportRenderer()
        md = renderer.render(report)
        print(md)


if __name__ == "__main__":
    main()
