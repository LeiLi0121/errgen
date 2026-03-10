"""
Run artifact persistence.

Saves every pipeline run to disk as a structured JSON directory so that:
  - Runs are fully reproducible and inspectable
  - Any sentence in the final report can be traced to its original evidence
  - Researchers can analyse failure modes, revision patterns, checker behavior

Directory layout per run
------------------------
runs/{run_id}/
  manifest.json         – run metadata (status, timestamps, config snapshot)
  request.json          – parsed UserRequest
  sources.json          – SourceMetadata list
  evidence_chunks.json  – all EvidenceChunk objects
  extracted_facts.json  – all ExtractedFact objects
  calc_requests.json    – CalculationRequest list
  calc_results.json     – CalculationResult list
  paragraphs/
    {paragraph_id}.json – each AnalysisParagraph with full revision history
  verdicts/
    {verdict_id}.json   – each CheckerVerdict
  revisions/
    {revision_id}.json  – each RevisionRecord
  report.json           – FinalReport (full)
  report.md             – Rendered Markdown report
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from errgen.config import Config
from errgen.models import RunArtifact

logger = logging.getLogger(__name__)


class RunRecord:
    """Persists a RunArtifact to a structured directory."""

    def __init__(self, runs_dir: str | None = None) -> None:
        self.runs_dir = Path(runs_dir or Config.RUNS_DIR)

    def save(self, artifact: RunArtifact) -> Path:
        """
        Persist the run artifact to disk.

        Returns the path to the run directory.
        Does NOT raise on IO errors (logs a warning instead) so that a
        persistence failure never crashes the pipeline.
        """
        run_dir = self.runs_dir / artifact.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._save_manifest(artifact, run_dir)
            self._save_json("request.json", artifact.request.model_dump(mode="json"), run_dir)
            self._save_json(
                "sources.json",
                [s.model_dump(mode="json") for s in artifact.retrieved_sources],
                run_dir,
            )
            self._save_json(
                "evidence_chunks.json",
                [c.model_dump(mode="json") for c in artifact.evidence_chunks],
                run_dir,
            )
            self._save_json(
                "extracted_facts.json",
                [f.model_dump(mode="json") for f in artifact.extracted_facts],
                run_dir,
            )
            self._save_json(
                "calc_requests.json",
                [r.model_dump(mode="json") for r in artifact.calculation_requests],
                run_dir,
            )
            self._save_json(
                "calc_results.json",
                [r.model_dump(mode="json") for r in artifact.calculation_results],
                run_dir,
            )

            # Per-paragraph JSON files
            para_dir = run_dir / "paragraphs"
            para_dir.mkdir(exist_ok=True)
            all_paragraphs = list(artifact.paragraph_drafts)
            if artifact.final_report:
                for section in artifact.final_report.sections:
                    all_paragraphs.extend(section.paragraphs)
            seen_ids: set[str] = set()
            for para in all_paragraphs:
                if para.paragraph_id in seen_ids:
                    continue
                seen_ids.add(para.paragraph_id)
                self._save_json(
                    f"{para.paragraph_id}.json",
                    para.model_dump(mode="json"),
                    para_dir,
                )

            # Per-verdict JSON files
            verdict_dir = run_dir / "verdicts"
            verdict_dir.mkdir(exist_ok=True)
            for verdict in artifact.checker_verdicts:
                self._save_json(
                    f"{verdict.verdict_id}.json",
                    verdict.model_dump(mode="json"),
                    verdict_dir,
                )

            # Per-revision JSON files
            revision_dir = run_dir / "revisions"
            revision_dir.mkdir(exist_ok=True)
            for revision in artifact.revision_records:
                self._save_json(
                    f"{revision.revision_id}.json",
                    revision.model_dump(mode="json"),
                    revision_dir,
                )

            # Final report
            if artifact.final_report:
                self._save_json(
                    "report.json",
                    artifact.final_report.model_dump(mode="json"),
                    run_dir,
                )
                # Rendered Markdown
                try:
                    from errgen.report import ReportRenderer
                    renderer = ReportRenderer()
                    md = renderer.render(artifact.final_report)
                    (run_dir / "report.md").write_text(md, encoding="utf-8")
                except Exception as exc:
                    logger.warning("Could not render Markdown report: %s", exc)

            logger.info("Run artifact saved to %s", run_dir)

        except Exception as exc:
            logger.error(
                "Failed to save run artifact %s: %s", artifact.run_id, exc
            )

        return run_dir

    # ------------------------------------------------------------------

    @staticmethod
    def _save_manifest(artifact: RunArtifact, run_dir: Path) -> None:
        manifest = {
            "run_id": artifact.run_id,
            "ticker": artifact.request.ticker,
            "as_of_date": artifact.request.as_of_date,
            "status": artifact.status,
            "started_at": artifact.started_at.isoformat(),
            "completed_at": (
                artifact.completed_at.isoformat() if artifact.completed_at else None
            ),
            "n_sources": len(artifact.retrieved_sources),
            "n_chunks": len(artifact.evidence_chunks),
            "n_facts": len(artifact.extracted_facts),
            "n_calculations": len(artifact.calculation_results),
            "n_paragraphs": len(artifact.paragraph_drafts),
            "n_verdicts": len(artifact.checker_verdicts),
            "n_revisions": len(artifact.revision_records),
            "errors": artifact.errors,
            "warnings": artifact.warnings,
            "overall_report_status": (
                artifact.final_report.overall_status.value
                if artifact.final_report
                else None
            ),
            "config_snapshot": Config.as_dict(),
        }
        RunRecord._save_json("manifest.json", manifest, run_dir)

    @staticmethod
    def _save_json(filename: str, data: object, directory: Path) -> None:
        path = directory / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    def load(self, run_id: str) -> dict:
        """
        Load the manifest for a previous run.
        Useful for comparing runs or resuming analysis.
        """
        manifest_path = self.runs_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No run found with id: {run_id}")
        with open(manifest_path, encoding="utf-8") as fh:
            return json.load(fh)

    def list_runs(self) -> list[dict]:
        """Return a summary list of all recorded runs, newest first."""
        if not self.runs_dir.exists():
            return []
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as fh:
                        runs.append(json.load(fh))
                except Exception:
                    pass
        return runs
