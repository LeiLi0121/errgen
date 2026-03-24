#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PAGE_ONE_SECTION_HEADINGS = [
    "Company Overview",
    "Investment Overview",
    "Risks",
]

SKIP_LINE_PATTERNS = [
    re.compile(r"^DBS Group Research$"),
    re.compile(r"^Disclaimer: The information contained in this document"),
    re.compile(r".*prior written consent.*"),
    re.compile(r".*Please refer to Disclaimer found at the end of this document.*"),
    re.compile(r".*found at the end of this document\.*"),
]

STOP_PAGE_ONE_MARKERS = {
    "Analysts",
    "Key Financial Data",
    "Bloomberg Ticker",
    "Indexed Share Price vs Composite Index Performance",
    "Source: Bloomberg",
    "Financial Summary",
}


@dataclass
class ReportRecord:
    report_id: str
    company: str
    report_date: str | None
    headline: str
    rating: str | None
    target_price: float | None
    target_currency: str | None
    narrative_text: str
    full_text_path: str
    narrative_text_path: str
    minimal_prompt: str
    sections: dict[str, str]


def run_pdftotext(pdf_path: Path, first_page_only: bool = False) -> str:
    cmd = ["pdftotext", "-raw"]
    if first_page_only:
        cmd.extend(["-f", "1", "-l", "1"])
    cmd.extend([str(pdf_path), "-"])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in normalize_text(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if any(pattern.match(line) for pattern in SKIP_LINE_PATTERNS):
            continue
        lines.append(line)
    return lines


def parse_page_one(lines: list[str]) -> tuple[str, str | None, str, dict[str, str]]:
    company_overview_idx = lines.index("Company Overview")
    header_lines = [
        line for line in lines[:company_overview_idx] if line != "US EQUITY RESEARCH"
    ]

    company, report_date, headline = parse_header_block(header_lines)

    body_lines: list[str] = []
    for line in lines[company_overview_idx + 1 :]:
        if line in STOP_PAGE_ONE_MARKERS:
            break
        body_lines.append(line)

    sections: dict[str, list[str]] = {"Company Overview": []}
    current: str | None = "Company Overview"
    for line in body_lines:
        if line in PAGE_ONE_SECTION_HEADINGS:
            current = line
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        sections[current].append(line)

    joined_sections = {
        name: collapse_wrapped_lines(values)
        for name, values in sections.items()
        if values
    }
    return company, report_date, headline, joined_sections


def parse_header_block(header_lines: list[str]) -> tuple[str, str | None, str]:
    if len(header_lines) < 2:
        raise ValueError("Could not locate page-one header block.")

    if len(header_lines) >= 3 and is_date_only_line(header_lines[0]):
        report_date = extract_date(header_lines[0])
        return header_lines[1], report_date, header_lines[2]

    if len(header_lines) >= 3 and is_date_only_line(header_lines[1]):
        report_date = extract_date(header_lines[1])
        return header_lines[2], report_date, header_lines[3] if len(header_lines) >= 4 else ""

    if len(header_lines) >= 3 and is_date_only_line(header_lines[-1]):
        report_date = extract_date(header_lines[-1])
        return header_lines[-3], report_date, header_lines[-2]

    if len(header_lines) >= 2 and line_contains_date(header_lines[-1]):
        report_date = extract_date(header_lines[-1])
        headline = remove_date_from_line(header_lines[-1]).strip()
        company = header_lines[-2]
        if not headline:
            raise ValueError("Could not parse headline from header line.")
        return company, report_date, headline

    if len(header_lines) >= 3 and is_date_only_line(header_lines[-3]):
        report_date = extract_date(header_lines[-3])
        return header_lines[-2], report_date, header_lines[-1]

    raise ValueError("Could not locate page-one date/company/headline block.")


def collapse_wrapped_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


DATE_PATTERNS = [
    ("%d %b %Y", re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4}\b")),
    ("%d %B %Y", re.compile(r"\b\d{1,2} [A-Z][a-z]+ \d{4}\b")),
]


def is_date_only_line(line: str) -> bool:
    return extract_date(line) is not None and remove_date_from_line(line).strip() == ""


def line_contains_date(line: str) -> bool:
    return extract_date(line) is not None


def extract_date(line: str) -> str | None:
    for fmt, pattern in DATE_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(0), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def remove_date_from_line(line: str) -> str:
    for _, pattern in DATE_PATTERNS:
        line = pattern.sub("", line)
    return " ".join(line.split())


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_focus_phrases(headline: str, investment_text: str) -> list[str]:
    focus_phrases: list[str] = []
    if headline:
        focus_phrases.append(headline.rstrip("."))

    line_leads = re.findall(r"(?m)(?:^|(?<=\n))([A-Z][^.]{8,120}\.)", investment_text)
    for lead in line_leads:
        focus_phrases.append(lead.strip())

    if len(focus_phrases) == 1:
        focus_phrases.extend(sentence_split(investment_text)[:3])

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in focus_phrases:
        norm = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(phrase.rstrip("."))
    return deduped[:4]


def oxford_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def build_minimal_prompt(company: str, report_date: str | None, headline: str, sections: dict[str, str]) -> str:
    date_clause = f" as of {report_date}" if report_date else ""
    focuses = extract_focus_phrases(headline, sections.get("Investment Overview", ""))
    focus_text = oxford_join([focus.rstrip(" .") for focus in focuses]).rstrip(" .")
    focus_clause = f" focusing on {focus_text}" if focus_text else ""
    risk_clause = " Include the key risks discussed in the source report."
    return f"Write an equity research report for {company}{date_clause}{focus_clause}.{risk_clause}"


def extract_rating_and_target(text: str) -> tuple[str | None, float | None, str | None]:
    rating_match = re.search(
        r"\bDBS Rating\s+(STRONG BUY|BUY|HOLD|FULLY VALUED|SELL)\b",
        text,
    )
    rating = rating_match.group(1) if rating_match else None

    target_patterns = [
        re.compile(
            r"\b12-mth Target Price \(([A-Z]{3})\)\s+([0-9][0-9,]*(?:\.[0-9]+)?)\b",
            re.I,
        ),
        re.compile(
            r"\bTP(?: for [A-Z]+)?(?: is)?(?: based on)?(?: a higher)?(?: revised)?(?: to| of) ([A-Z]{3})\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b",
            re.I,
        ),
        re.compile(
            r"\btarget price(?: of| to)? ([A-Z]{3})\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b",
            re.I,
        ),
    ]
    target_price = None
    target_currency = None
    for pattern in target_patterns:
        match = pattern.search(text)
        if not match:
            continue
        target_currency = match.group(1).upper()
        target_price = float(match.group(2).replace(",", ""))
        break

    return rating, target_price, target_currency


def clean_full_text(text: str) -> str:
    lines = clean_lines(text)
    cleaned: list[str] = []
    for line in lines:
        if line.startswith("Source: Bloomberg"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def build_record(pdf_path: Path, output_dir: Path) -> ReportRecord:
    report_id = pdf_path.stem
    full_text_raw = run_pdftotext(pdf_path, first_page_only=False)
    page_one_raw = run_pdftotext(pdf_path, first_page_only=True)

    full_text = clean_full_text(full_text_raw)
    page_one_lines = clean_lines(page_one_raw)
    company, report_date, headline, sections = parse_page_one(page_one_lines)
    rating, target_price, target_currency = extract_rating_and_target(full_text)

    narrative_chunks = [headline]
    for heading in PAGE_ONE_SECTION_HEADINGS:
        body = sections.get(heading)
        if body:
            narrative_chunks.append(f"{heading}: {body}")
    narrative_text = "\n\n".join(narrative_chunks).strip() + "\n"

    full_text_path = output_dir / "full_text" / f"{report_id}.txt"
    narrative_text_path = output_dir / "narrative_text" / f"{report_id}.txt"
    full_text_path.parent.mkdir(parents=True, exist_ok=True)
    narrative_text_path.parent.mkdir(parents=True, exist_ok=True)
    full_text_path.write_text(full_text, encoding="utf-8")
    narrative_text_path.write_text(narrative_text, encoding="utf-8")

    return ReportRecord(
        report_id=report_id,
        company=company,
        report_date=report_date,
        headline=headline,
        rating=rating,
        target_price=target_price,
        target_currency=target_currency,
        narrative_text=narrative_text,
        full_text_path=str(full_text_path),
        narrative_text_path=str(narrative_text_path),
        minimal_prompt=build_minimal_prompt(company, report_date, headline, sections),
        sections=sections,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract clean text and minimal prompts from PDF reports.")
    parser.add_argument("--input-dir", default="reports", help="Directory containing PDF reports.")
    parser.add_argument(
        "--output-dir",
        default="reports_extracted",
        help="Directory to write extracted text and metadata.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {input_dir}")

    records: list[ReportRecord] = []
    for pdf_path in pdf_files:
        records.append(build_record(pdf_path, output_dir))

    jsonl_path = output_dir / "reports.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {
                        "report_id": record.report_id,
                        "company": record.company,
                        "report_date": record.report_date,
                        "headline": record.headline,
                        "rating": record.rating,
                        "target_price": record.target_price,
                        "target_currency": record.target_currency,
                        "sections": record.sections,
                        "full_text_path": record.full_text_path,
                        "narrative_text_path": record.narrative_text_path,
                        "minimal_prompt": record.minimal_prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
