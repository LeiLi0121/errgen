"""
Rerun only the prediction stage for an existing ERRGen run.

This script reuses the saved upstream artifacts from a completed run:
  - verified / drafted analysis sections
  - evidence chunks
  - calculation results

It strips the existing "Investment Recommendation & Outlook" section, reruns
either the full `prediction_agent` or `baseline_prediction_agent`, and then
reassembles a fresh report via `report_writer`.

Typical usage:

python3 scripts/rerun_prediction_from_run.py \
  --run-dir evaluation/runs/full/1227f39c-b18f-4c19-8caf-d6c20ba81fb9 \
  --mode full \
  --output-root evaluation/runs/full_prediction_reruns
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from errgen.config import Config
from errgen.models import (
    CalculationRequest,
    CalculationResult,
    CheckerVerdict,
    EvidenceChunk,
    ExtractedFact,
    FinalReport,
    ReportSection,
    RevisionRecord,
    RunArtifact,
    SourceMetadata,
    UserRequest,
)
from errgen.nodes.baseline_prediction_agent import baseline_prediction_agent
from errgen.nodes.prediction_agent import prediction_agent
from errgen.report import ReportAssembler, ReportRenderer
from errgen.run_record import RunRecord

PREDICTION_SECTION_NAME = "Investment Recommendation & Outlook"
APPEND_ONLY_STATE_KEYS = {
    "paragraph_drafts",
    "checker_verdicts",
    "revision_records",
    "warnings",
    "errors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun only the prediction stage for existing ERRGen runs.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Path to a completed run directory. Repeatable.",
    )
    parser.add_argument(
        "--runs-root",
        help="Optional root containing run directories. Use with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every immediate child run directory under --runs-root.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "full", "baseline"],
        help="Prediction mode to rerun. Default infers from the run path.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Where to write rerun artifacts. Defaults to the source run's parent directory.",
    )
    parser.add_argument(
        "--suffix",
        default="pred-rerun",
        help="Suffix used when generating new run ids.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of concurrent rerun workers.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other runs if one rerun fails.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(path: Path, model_cls):
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def _load_model_list(path: Path, model_cls) -> list:
    return [model_cls.model_validate(item) for item in _load_json(path)]


def _infer_mode(run_dir: Path, manifest: dict) -> str:
    parent_name = run_dir.parent.name.lower()
    if parent_name in {"full", "baseline"}:
        return parent_name
    if manifest.get("n_verdicts", 0) or manifest.get("n_revisions", 0):
        return "full"
    return "baseline"


def _collect_run_dirs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [Path(item).resolve() for item in args.run_dir]
    if args.runs_root and args.all:
        root = Path(args.runs_root).resolve()
        run_dirs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in run_dirs:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    if not deduped:
        raise SystemExit("No run directories provided. Use --run-dir or --runs-root with --all.")
    return deduped


def _non_prediction_sections(report: FinalReport) -> list[ReportSection]:
    return [
        section
        for section in report.sections
        if section.section_name != PREDICTION_SECTION_NAME
    ]


def _load_filtered_verdicts(run_dir: Path, keep_paragraph_ids: set[str]) -> list[CheckerVerdict]:
    verdict_dir = run_dir / "verdicts"
    if not verdict_dir.exists():
        return []
    verdicts: list[CheckerVerdict] = []
    for path in sorted(verdict_dir.glob("*.json")):
        verdict = _load_model(path, CheckerVerdict)
        if verdict.paragraph_id in keep_paragraph_ids:
            verdicts.append(verdict)
    return verdicts


def _load_filtered_revisions(run_dir: Path, keep_paragraph_ids: set[str]) -> list[RevisionRecord]:
    revision_dir = run_dir / "revisions"
    if not revision_dir.exists():
        return []
    revisions: list[RevisionRecord] = []
    for path in sorted(revision_dir.glob("*.json")):
        revision = _load_model(path, RevisionRecord)
        if revision.paragraph_id in keep_paragraph_ids:
            revisions.append(revision)
    return revisions


def _build_state(run_dir: Path, run_suffix: str) -> tuple[dict, str]:
    manifest = _load_json(run_dir / "manifest.json")
    request = _load_model(run_dir / "request.json", UserRequest)
    report = _load_model(run_dir / "report.json", FinalReport)
    sources = _load_model_list(run_dir / "sources.json", SourceMetadata)
    evidence_chunks = _load_model_list(run_dir / "evidence_chunks.json", EvidenceChunk)
    extracted_facts = _load_model_list(run_dir / "extracted_facts.json", ExtractedFact)
    calc_requests = _load_model_list(run_dir / "calc_requests.json", CalculationRequest)
    calc_results = _load_model_list(run_dir / "calc_results.json", CalculationResult)

    base_sections = _non_prediction_sections(report)
    keep_paragraph_ids = {
        para.paragraph_id
        for section in base_sections
        for para in section.paragraphs
    }

    upstream_verdicts = _load_filtered_verdicts(run_dir, keep_paragraph_ids)
    upstream_revisions = _load_filtered_revisions(run_dir, keep_paragraph_ids)

    # Keep only upstream warnings. Old prediction warnings should not leak into the rerun.
    warnings = [
        warning
        for warning in (manifest.get("warnings") or [])
        if "Prediction section" not in warning
    ]

    new_run_id = (
        f"{run_dir.name}-{run_suffix}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    state = {
        "query": request.raw_text,
        "ticker": request.ticker,
        "company_name": request.company_name or "",
        "focus": list(request.focus_areas),
        "as_of_date": request.as_of_date,
        "run_id": new_run_id,
        "sources": sources,
        "evidence_chunks": evidence_chunks,
        "extracted_facts": extracted_facts,
        "calculation_requests": calc_requests,
        "calculations": calc_results,
        "analysis_sections": base_sections,
        "paragraph_drafts": [],
        "checker_verdicts": upstream_verdicts,
        "revision_records": upstream_revisions,
        "revision_count": 0,
        "has_blocking_issues": False,
        "warnings": warnings,
        "errors": list(manifest.get("errors") or []),
        "report": None,
        "report_md": None,
    }

    return state, new_run_id


def rerun_prediction_for_run(
    run_dir: Path,
    mode: str,
    output_root: Path,
    run_suffix: str,
) -> Path:
    state, run_id = _build_state(run_dir, run_suffix)

    if mode == "full":
        prediction_update = prediction_agent(state)
    elif mode == "baseline":
        prediction_update = baseline_prediction_agent(state)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for key, value in prediction_update.items():
        if key in APPEND_ONLY_STATE_KEYS:
            state[key] = list(state.get(key) or []) + list(value or [])
        else:
            state[key] = value
    state.update(_assemble_and_persist_report(state, output_root))
    return output_root / run_id


def _assemble_and_persist_report(state: dict, output_root: Path) -> dict:
    sections: list[ReportSection] = state["analysis_sections"]
    all_chunks: list[EvidenceChunk] = state["evidence_chunks"]
    calc_results: list[CalculationResult] = state["calculations"]
    warnings: list[str] = state.get("warnings") or []
    run_id: str = state["run_id"]

    request = UserRequest(
        raw_text=state.get("query", ""),
        ticker=state["ticker"],
        company_name=state.get("company_name"),
        as_of_date=state.get("as_of_date"),
        focus_areas=state.get("focus") or [],
    )

    assembler = ReportAssembler()
    report: FinalReport = assembler.assemble(
        request=request,
        sections=sections,
        all_chunks=all_chunks,
        calc_results=calc_results,
        warnings=list(warnings),
    )

    renderer = ReportRenderer()
    report_md = renderer.render(report)

    artifact = RunArtifact(
        run_id=run_id,
        request=request,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
        retrieved_sources=state.get("sources") or [],
        evidence_chunks=state.get("evidence_chunks") or [],
        extracted_facts=state.get("extracted_facts") or [],
        calculation_requests=state.get("calculation_requests") or [],
        calculation_results=state.get("calculations") or [],
        paragraph_drafts=state.get("paragraph_drafts") or [],
        checker_verdicts=state.get("checker_verdicts") or [],
        revision_records=state.get("revision_records") or [],
        final_report=report,
        warnings=state.get("warnings") or [],
        errors=state.get("errors") or [],
    )
    RunRecord(runs_dir=str(output_root)).save(artifact)

    return {
        "report": report,
        "report_md": report_md,
    }


def main() -> None:
    args = parse_args()
    run_dirs = _collect_run_dirs(args)
    Config.validate_required()

    tasks: list[tuple[Path, str, Path]] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        manifest = _load_json(run_dir / "manifest.json")
        mode = args.mode if args.mode != "auto" else _infer_mode(run_dir, manifest)
        output_root = Path(args.output_root).resolve() if args.output_root else run_dir.parent.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        tasks.append((run_dir, mode, output_root))

    max_workers = max(1, min(args.max_workers, len(tasks)))
    future_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for run_dir, mode, output_root in tasks:
            print(f"[queued] source={run_dir} mode={mode} output_root={output_root}")
            future = executor.submit(
                rerun_prediction_for_run,
                run_dir,
                mode,
                output_root,
                args.suffix,
            )
            future_map[future] = run_dir

        for future in as_completed(future_map):
            source_run_dir = future_map[future]
            try:
                new_run_dir = future.result()
                print(f"[done] source={source_run_dir} new_run={new_run_dir}")
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[error] source={source_run_dir} error={exc}")


if __name__ == "__main__":
    main()
