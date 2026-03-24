"""
I/O helpers for ERRGen evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evaluation.schemas import EvalQuery, RunBundle
from errgen.models import CalculationResult, EvidenceChunk, FinalReport, UserRequest


def load_queries(path: str | Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            queries.append(EvalQuery.model_validate_json(line))
    return queries


def load_run_bundle(run_dir: str | Path) -> RunBundle:
    run_path = Path(run_dir)
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    request = UserRequest.model_validate_json(
        (run_path / "request.json").read_text(encoding="utf-8")
    )
    report = FinalReport.model_validate_json(
        (run_path / "report.json").read_text(encoding="utf-8")
    )
    evidence_chunks = [
        EvidenceChunk.model_validate(item)
        for item in json.loads((run_path / "evidence_chunks.json").read_text(encoding="utf-8"))
    ]
    calculation_results = [
        CalculationResult.model_validate(item)
        for item in json.loads((run_path / "calc_results.json").read_text(encoding="utf-8"))
    ]
    report_text = (run_path / "report.md").read_text(encoding="utf-8")
    return RunBundle(
        run_dir=run_path,
        run_id=run_path.name,
        request=request,
        report=report,
        report_text=report_text,
        evidence_chunks=evidence_chunks,
        calculation_results=calculation_results,
        manifest=manifest,
    )


def index_runs_by_query(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    indexed: dict[str, tuple[datetime, Path]] = {}
    if not root.exists():
        return {}

    for child in root.iterdir():
        if not child.is_dir():
            continue
        request_path = child / "request.json"
        manifest_path = child / "manifest.json"
        if not request_path.exists() or not manifest_path.exists():
            continue

        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query = request_data.get("raw_text")
        if not query:
            continue

        completed_at = manifest.get("completed_at") or manifest.get("started_at")
        try:
            sort_key = datetime.fromisoformat(completed_at)
        except Exception:
            sort_key = datetime.fromtimestamp(child.stat().st_mtime)

        previous = indexed.get(query)
        if previous is None or sort_key > previous[0]:
            indexed[query] = (sort_key, child)

    return {query: path for query, (_, path) in indexed.items()}


def filter_queries(
    queries: list[EvalQuery],
    report_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[EvalQuery]:
    filtered = [
        item for item in queries
        if not report_ids or item.report_id in report_ids
    ]
    if limit is not None:
        return filtered[:limit]
    return filtered
