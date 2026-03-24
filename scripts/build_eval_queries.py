"""
Build ERRGen evaluation queries from extracted analyst-report narratives.

This script is designed for the workflow:

  narrative_text/*.txt  --(LLM)-> neutral ERRGen query
  rules / metadata      --(merge)-> report_id, paths, optional company/date
  output                -> JSONL ready for batch ERRGen runs

Typical usage
-------------

python scripts/build_eval_queries.py \
    --narrative-dir ../reports_extracted/narrative_text \
    --metadata-jsonl ../reports_extracted/reports.jsonl \
    --output ../reports_extracted/errgen_queries.jsonl \
    --model qwen-max

If your Qwen endpoint is OpenAI-compatible, you can use the hardcoded defaults
or pass overrides:

python scripts/build_eval_queries.py \
    --base-url https://your-qwen-endpoint/v1 \
    --api-key YOUR_KEY \
    --model qwen-plus
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

_workspace_root = Path(__file__).resolve().parent.parent


logger = logging.getLogger("errgen.build_eval_queries")

DEFAULT_OUTPUT_NAME = "errgen_queries.jsonl"
DEFAULT_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_API_KEY = "sk-1835a85c30164139a0cc0b47c2302a81"
DEFAULT_OPENAI_MODEL = "qwen3.5-plus"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_RETRY_ATTEMPTS = 3

SYSTEM_PROMPT = """You convert analyst report narrative plus rule-extracted metadata into a neutral ERRGen input query.

Return ONLY valid JSON with this exact schema:
{
  "query": "..."
}

Requirements:
- The query must be a reusable research request, not a summary of the answer.
- Keep the query neutral and evaluation-safe.
- Use the provided company_name, ticker, and as_of_date metadata when available.
- Ask for these analysis dimensions when relevant: financial performance, recent developments/news, company and industry positioning, key risks, and investment view.
- Do NOT include target price, final analyst rating, or direct analyst conclusions as requested outputs.
- Do NOT mention DBS, analysts, broker names, or source-document wording like "according to this report".
- Keep it concise, typically 1-2 sentences.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JSONL ERRGen eval queries from narrative report text.",
    )
    parser.add_argument(
        "--narrative-dir",
        default=str(_workspace_root / "reports_extracted" / "narrative_text"),
        help="Directory containing one narrative .txt file per report.",
    )
    parser.add_argument(
        "--full-text-dir",
        default=str(_workspace_root / "reports_extracted" / "full_text"),
        help="Directory containing full-text .txt files; used for path bookkeeping only.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(_workspace_root / "reports"),
        help="Directory containing source PDFs; used for path bookkeeping only.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        default=str(_workspace_root / "reports_extracted" / "reports.jsonl"),
        help="Optional JSONL with existing report metadata.",
    )
    parser.add_argument(
        "--output",
        default=str(_workspace_root / "reports_extracted" / DEFAULT_OUTPUT_NAME),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="LLM model override, e.g. qwen-max.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OPENAI_BASE_URL,
        help="OpenAI-compatible API base URL override.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_OPENAI_API_KEY,
        help="API key override for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum narrative characters sent to the LLM per report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N reports after sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chat_json(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error: Exception | None = None
    for attempt in range(DEFAULT_RETRY_ATTEMPTS):
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "LLM request failed (attempt %d/%d): %s",
                attempt + 1,
                DEFAULT_RETRY_ATTEMPTS,
                exc,
            )
            continue

        try:
            content = raw["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON response payload from LLM: {raw}") from exc

    raise RuntimeError(
        f"LLM request failed after {DEFAULT_RETRY_ATTEMPTS} attempts. Last error: {last_error}"
    )


def load_metadata_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        logger.info("Metadata JSONL not found at %s; continuing without it.", path)
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            report_id = str(record.get("report_id") or "").strip()
            if report_id:
                metadata[report_id] = record
    logger.info("Loaded %d metadata rows from %s", len(metadata), path)
    return metadata


def list_narratives(directory: Path, limit: int | None) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Narrative directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if limit is not None:
        files = files[:limit]
    return files


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def clean_narrative(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    text = re.sub(r"[ \t]+", " ", text)
    return text[:max_chars].strip()


def infer_company_name(
    report_id: str,
    metadata: dict[str, Any] | None,
    full_text: str,
) -> str | None:
    metadata = metadata or {}
    company = str(metadata.get("company") or "").strip()
    if company:
        return company

    lines = [line.strip() for line in full_text.splitlines()[:12] if line.strip()]
    for idx, line in enumerate(lines):
        if line.upper() == "US EQUITY RESEARCH":
            continue
        if re.search(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", line):
            continue
        if line.lower() in {"company overview", "investment overview", "risks"}:
            continue
        if idx + 1 < len(lines) and "company overview" in lines[idx + 1].lower():
            return line

    return report_id.replace("_", " ")


def infer_ticker(metadata: dict[str, Any] | None, full_text: str) -> str | None:
    metadata = metadata or {}

    # 1) Bloomberg ticker in the report body
    bloomberg_match = re.search(r"Bloomberg Ticker\s+([A-Z0-9.\-]+(?:\s+[A-Z]{2,4})?)", full_text)
    if bloomberg_match:
        return bloomberg_match.group(1).strip()

    # 2) Company name with exchange ticker in brackets, e.g. (NYSE: AXP)
    exchange_match = re.search(r"\((?:NYSE|NASDAQ|Nasdaq|NYSEARCA|NYSE American):\s*([A-Z0-9.\-]+)\)", full_text)
    if exchange_match:
        return exchange_match.group(1).strip()

    # 3) Plain company-name ticker in first page text, e.g. Alphabet Inc. (GOOGL)
    paren_match = re.search(r"\(([A-Z]{1,6}(?:\.[A-Z]{1,4})?)\)", "\n".join(full_text.splitlines()[:30]))
    if paren_match:
        return paren_match.group(1).strip()

    for key in ("ticker", "bloomberg_ticker"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return None


def _parse_date_token(raw: str) -> str | None:
    raw = raw.strip().replace(",", "")
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m":
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")
    return None


def infer_as_of_date(metadata: dict[str, Any] | None, full_text: str) -> str | None:
    metadata = metadata or {}
    report_date = str(metadata.get("report_date") or "").strip()
    if report_date:
        normalized = _parse_date_token(report_date)
        if normalized:
            return normalized

    lines = [line.strip() for line in full_text.splitlines()[:12] if line.strip()]
    patterns = (
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}-\d{2}\b",
    )
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            normalized = _parse_date_token(match.group(0))
            if normalized:
                return normalized
    return None


def build_user_prompt(
    report_id: str,
    narrative_text: str,
    company_name: str | None,
    ticker: str | None,
    as_of_date: str | None,
    metadata: dict[str, Any] | None,
) -> str:
    metadata = metadata or {}
    metadata_block = {
        "report_id": report_id,
        "company_name": company_name,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "headline": metadata.get("headline"),
    }
    return (
        f"Metadata:\n{json.dumps(metadata_block, ensure_ascii=False)}\n\n"
        f"Narrative:\n{narrative_text}\n\n"
        "Write a neutral query that can be fed into ERRGen to regenerate a comparable equity research report.\n"
        "The query should request analysis, not restate the source report's final answer."
    )


def generate_query(
    report_id: str,
    narrative_text: str,
    company_name: str | None,
    ticker: str | None,
    as_of_date: str | None,
    metadata: dict[str, Any] | None,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    messages = build_messages(
        SYSTEM_PROMPT,
        build_user_prompt(
            report_id=report_id,
            narrative_text=narrative_text,
            company_name=company_name,
            ticker=ticker,
            as_of_date=as_of_date,
            metadata=metadata,
        ),
    )
    raw = chat_json(
        messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=300,
    )
    query = str(raw.get("query") or "").strip()
    if not query:
        raise ValueError(f"LLM returned empty query for report {report_id}")
    return _normalize_query(query)


def _normalize_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    if query and query[-1] not in ".!?":
        query += "."
    return query


def build_record(
    report_id: str,
    narrative_path: Path,
    full_text_dir: Path,
    pdf_dir: Path,
    company_name: str | None,
    ticker: str | None,
    as_of_date: str | None,
    query: str,
    metadata: dict[str, Any] | None,
    model: str,
) -> dict[str, Any]:
    metadata = metadata or {}
    full_text_path = full_text_dir / narrative_path.name
    pdf_path = pdf_dir / f"{report_id}.pdf"
    record: dict[str, Any] = {
        "report_id": report_id,
        "pdf_path": str(pdf_path.resolve()),
        "full_text_path": str(full_text_path.resolve()),
        "narrative_text_path": str(narrative_path.resolve()),
        "company_name": company_name,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "query": query,
        "llm_model": model,
    }

    for key in (
        "company",
        "report_date",
        "headline",
        "rating",
        "target_price",
        "target_currency",
    ):
        if metadata.get(key) is not None:
            record[key] = metadata[key]

    return record


def apply_runtime_overrides(args: argparse.Namespace) -> tuple[str, str, str]:
    base_url = args.base_url or DEFAULT_OPENAI_BASE_URL
    api_key = args.api_key or DEFAULT_OPENAI_API_KEY
    model = args.model or DEFAULT_OPENAI_MODEL
    if not api_key or not base_url:
        raise ValueError(
            "No API credentials configured. Pass --api-key / --base-url or edit the script defaults."
        )
    return base_url, api_key, model


def write_jsonl(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Re-run with --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    narrative_dir = Path(args.narrative_dir).resolve()
    full_text_dir = Path(args.full_text_dir).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    metadata_path = Path(args.metadata_jsonl).resolve()
    output_path = Path(args.output).resolve()

    base_url, api_key, model = apply_runtime_overrides(args)
    metadata_map = load_metadata_map(metadata_path)
    narratives = list_narratives(narrative_dir, args.limit)

    logger.info("Found %d narrative files in %s", len(narratives), narrative_dir)

    records: list[dict[str, Any]] = []
    for idx, narrative_path in enumerate(narratives, start=1):
        report_id = narrative_path.stem
        metadata = metadata_map.get(report_id)
        narrative_text = clean_narrative(read_text(narrative_path), args.max_chars)
        full_text_path = full_text_dir / narrative_path.name
        full_text = read_text(full_text_path) if full_text_path.exists() else ""
        if not narrative_text:
            logger.warning("Skipping empty narrative file: %s", narrative_path)
            continue

        company_name = infer_company_name(report_id=report_id, metadata=metadata, full_text=full_text)
        ticker = infer_ticker(metadata=metadata, full_text=full_text)
        as_of_date = infer_as_of_date(metadata=metadata, full_text=full_text)

        logger.info("[%d/%d] Generating query for %s", idx, len(narratives), report_id)
        query = generate_query(
            report_id=report_id,
            narrative_text=narrative_text,
            company_name=company_name,
            ticker=ticker,
            as_of_date=as_of_date,
            metadata=metadata,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        records.append(
            build_record(
                report_id=report_id,
                narrative_path=narrative_path,
                full_text_dir=full_text_dir,
                pdf_dir=pdf_dir,
                company_name=company_name,
                ticker=ticker,
                as_of_date=as_of_date,
                query=query,
                metadata=metadata,
                model=model,
            )
        )

    write_jsonl(output_path, records, overwrite=args.overwrite)
    logger.info("Wrote %d ERRGen eval queries to %s", len(records), output_path)


if __name__ == "__main__":
    main()
