#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You convert equity research report metadata into a neutral input
query for an evidence-grounded report generator.

Return ONLY valid JSON with these exact fields:
{
  "company_name": "Full company name",
  "ticker": "Ticker symbol used by the source report",
  "as_of_date": "YYYY-MM-DD or YYYY-MM or null",
  "query": "A single natural-language request for generating an equity research report"
}

Rules:
- Infer company_name, ticker, and as_of_date from the provided PDF excerpt.
- Prefer the exact ticker formatting used by the report when available.
- The query must be neutral and reusable.
- The query must ask for a full equity research report covering:
  financial performance, recent developments/news, company/market/industry positioning,
  key risks, and an investment recommendation.
- Do NOT include the source report's rating, target price, or final thesis in the query.
- Do NOT mention the source broker, analyst, or PDF.
- If the day is unavailable, use YYYY-MM. If month is also unavailable, use null.
"""


USER_PROMPT_TEMPLATE = """Generate a neutral errgen input query from this report excerpt.

File name: {file_name}
Report ID: {report_id}

Excerpt:
\"\"\"
{excerpt}
\"\"\"
"""


@dataclass
class QueryRecord:
    report_id: str
    file_name: str
    pdf_path: str
    text_path: str
    excerpt_path: str
    company_name: str
    ticker: str
    as_of_date: str | None
    query: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PDF text and build errgen queries with an OpenAI-compatible LLM."
    )
    parser.add_argument(
        "--input-dir",
        default="reports",
        help="Directory containing PDF reports.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports_errgen_queries",
        help="Directory for extracted text and generated query metadata.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "qwen-plus",
        help="Model name for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-pages",
        type=int,
        default=3,
        help="Number of leading PDF pages to send to the LLM.",
    )
    parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=12000,
        help="Maximum excerpt length sent to the LLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted text and JSONL output.",
    )
    return parser.parse_args()


def build_client(api_key: str | None, base_url: str | None) -> Any:
    if not api_key:
        raise SystemExit(
            "Missing API key. Set QWEN_API_KEY or OPENAI_API_KEY, or pass --api-key."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependency 'openai'. Install it in the environment "
            "that will run this script."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def run_pdftotext(pdf_path: Path, first_pages: int | None = None) -> str:
    cmd = ["pdftotext", "-raw"]
    if first_pages is not None:
        cmd.extend(["-f", "1", "-l", str(first_pages)])
    cmd.extend([str(pdf_path), "-"])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pdftotext failed for {pdf_path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def make_report_id(pdf_path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")
    return slug or "report"


def build_excerpt(full_or_partial_text: str, max_chars: int) -> str:
    excerpt = normalize_text(full_or_partial_text)
    return excerpt[:max_chars].strip()


def llm_extract_query(
    client: Any,
    model: str,
    report_id: str,
    file_name: str,
    excerpt: str,
) -> dict[str, Any]:
    try:
        from openai import APIConnectionError, APIError, RateLimitError
    except ImportError as exc:
        raise SystemExit(
            "Missing Python dependency 'openai'. Install it in the environment "
            "that will run this script."
        ) from exc

    user_prompt = USER_PROMPT_TEMPLATE.format(
        file_name=file_name,
        report_id=report_id,
        excerpt=excerpt,
    )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=600,
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_exc = exc
        except (RateLimitError, APIError, APIConnectionError) as exc:
            last_exc = exc
            time.sleep(2**attempt)

    raise RuntimeError(f"LLM extraction failed for {file_name}: {last_exc}")


def validate_payload(payload: dict[str, Any], pdf_path: Path) -> tuple[str, str, str | None, str]:
    company_name = str(payload.get("company_name") or "").strip()
    ticker = str(payload.get("ticker") or "").strip()
    as_of_date_raw = payload.get("as_of_date")
    as_of_date = str(as_of_date_raw).strip() if as_of_date_raw else None
    query = str(payload.get("query") or "").strip()

    if not company_name:
        raise ValueError(f"Missing company_name for {pdf_path.name}")
    if not ticker:
        raise ValueError(f"Missing ticker for {pdf_path.name}")
    if not query:
        raise ValueError(f"Missing query for {pdf_path.name}")

    return company_name, ticker, as_of_date, query


def write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    client: Any,
    model: str,
    llm_pages: int,
    max_excerpt_chars: int,
    overwrite: bool,
) -> QueryRecord:
    report_id = make_report_id(pdf_path)
    full_text = normalize_text(run_pdftotext(pdf_path))
    excerpt_text = build_excerpt(run_pdftotext(pdf_path, first_pages=llm_pages), max_excerpt_chars)

    text_path = output_dir / "text" / f"{report_id}.txt"
    excerpt_path = output_dir / "excerpt" / f"{report_id}.txt"
    write_text(text_path, full_text, overwrite=overwrite)
    write_text(excerpt_path, excerpt_text + "\n", overwrite=overwrite)

    payload = llm_extract_query(
        client=client,
        model=model,
        report_id=report_id,
        file_name=pdf_path.name,
        excerpt=excerpt_text,
    )
    company_name, ticker, as_of_date, query = validate_payload(payload, pdf_path)

    return QueryRecord(
        report_id=report_id,
        file_name=pdf_path.name,
        pdf_path=str(pdf_path.resolve()),
        text_path=str(text_path.resolve()),
        excerpt_path=str(excerpt_path.resolve()),
        company_name=company_name,
        ticker=ticker,
        as_of_date=as_of_date,
        query=query,
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {input_dir}")

    client = build_client(api_key=args.api_key, base_url=args.base_url)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[QueryRecord] = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...", file=sys.stderr)
        record = process_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            client=client,
            model=args.model,
            llm_pages=args.llm_pages,
            max_excerpt_chars=args.max_excerpt_chars,
            overwrite=args.overwrite,
        )
        records.append(record)

    jsonl_path = output_dir / "queries.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {jsonl_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
