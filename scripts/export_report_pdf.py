#!/usr/bin/env python3
"""
Render an existing ERRGen run directory to a styled PDF report.

This script is intentionally standalone: it reads a previously generated
run directory, so historical `runs/` and `evaluation/runs/` artifacts can be
exported without re-running the report pipeline.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "errgen_pdf_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepInFrame,
    LongTable,
    NextPageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageTemplate,
)


PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.55 * inch
TOP_MARGIN = 1.1 * inch
BOTTOM_MARGIN = 0.7 * inch
SIDEBAR_WIDTH = 2.45 * inch
GUTTER = 0.28 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
MAIN_WIDTH = CONTENT_WIDTH - SIDEBAR_WIDTH - GUTTER
CONTENT_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN

ACCENT_RED = colors.HexColor("#c53b31")
ACCENT_GOLD = colors.HexColor("#d4a72c")
INK = colors.HexColor("#111111")
MUTED = colors.HexColor("#6b6b6b")
LIGHT_BORDER = colors.HexColor("#d9d9d9")
LIGHT_FILL = colors.HexColor("#f4f4f4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an existing ERRGen run directory to PDF."
    )
    parser.add_argument(
        "run_dir",
        help="Path to an existing run directory containing report.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PDF path. Defaults to <run_dir>/report.pdf.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def rows_to_map(rows: list[list[str]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2:
            output[str(row[0])] = str(row[1])
    return output


def slug_to_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def markdown_to_paragraph_text(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = escaped.replace("\n", "<br/>")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", escaped)
    return escaped


def clean_text(text: str) -> str:
    text = re.sub(r"\[[CK]\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def format_metric_value(value: str | None) -> str:
    if not value:
        return "N/A"
    return slug_to_text(value)


def resolve_display_as_of_date(request: dict[str, Any]) -> str:
    as_of_date = str(request.get("as_of_date") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
        return as_of_date

    raw_text = str(request.get("raw_text") or "")
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_text)
    if iso_match:
        return iso_match.group(1)

    patterns = (
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if not match:
            continue
        token = match.group(0).replace(",", "")
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

    return as_of_date or "N/A"


def build_price_chart(run_dir: Path, output_dir: Path) -> Path | None:
    sources_path = run_dir / "sources.json"
    if not sources_path.exists():
        return None
    sources = load_json(sources_path)
    stock_source = next(
        (
            item
            for item in sources
            if item.get("source_type") == "price_data"
            and item.get("metadata", {}).get("series_role") == "stock"
        ),
        None,
    )
    benchmark_source = next(
        (
            item
            for item in sources
            if item.get("source_type") == "price_data"
            and item.get("metadata", {}).get("series_role") == "benchmark"
        ),
        None,
    )
    if not stock_source or not benchmark_source:
        return None

    def extract_series(source: dict[str, Any]) -> tuple[str, list[str], list[float]]:
        md = source.get("metadata", {})
        hist = md.get("historical") or []
        dates: list[str] = []
        closes: list[float] = []
        for row in hist:
            close = row.get("close")
            date = row.get("date")
            if date is None or close is None:
                continue
            dates.append(str(date))
            closes.append(float(close))
        label = str(md.get("symbol") or source.get("ticker") or md.get("series_role") or "Series")
        return label, dates, closes

    stock_label, stock_dates, stock_closes = extract_series(stock_source)
    benchmark_label, benchmark_dates, benchmark_closes = extract_series(benchmark_source)
    if not stock_dates or not benchmark_dates:
        return None

    common_dates = sorted(set(stock_dates) & set(benchmark_dates))
    if len(common_dates) < 10:
        return None

    stock_map = dict(zip(stock_dates, stock_closes))
    benchmark_map = dict(zip(benchmark_dates, benchmark_closes))
    stock_series = [stock_map[d] for d in common_dates]
    benchmark_series = [benchmark_map[d] for d in common_dates]
    if not stock_series[0] or not benchmark_series[0]:
        return None

    stock_norm = [value / stock_series[0] * 100 for value in stock_series]
    benchmark_norm = [value / benchmark_series[0] * 100 for value in benchmark_series]

    fig, ax = plt.subplots(figsize=(4.0, 2.2), dpi=160)
    ax.plot(common_dates, stock_norm, color="#cf2e2e", linewidth=1.8, label=stock_label)
    ax.plot(common_dates, benchmark_norm, color="#d4a72c", linewidth=1.8, label=benchmark_label)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=6, colors="#666666", length=0)
    ax.tick_params(axis="y", labelsize=6, colors="#666666", length=0)
    ticks = [0, len(common_dates) // 3, (2 * len(common_dates)) // 3, len(common_dates) - 1]
    tick_labels = [common_dates[i][2:] for i in ticks]
    ax.set_xticks(ticks, tick_labels)
    ax.legend(loc="upper left", fontsize=6, frameon=False, ncol=2)
    ax.set_title("Indexed Share Price vs Benchmark", loc="left", fontsize=8, color="#666666", pad=6)
    fig.tight_layout(pad=0.8)

    output_path = output_dir / "price_chart.png"
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PdfTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=29,
            leading=33,
            textColor=INK,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "PdfSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "section": ParagraphStyle(
            "PdfSection",
            parent=styles["Heading2"],
            fontName="Helvetica",
            fontSize=13,
            leading=16,
            textColor=ACCENT_RED,
            spaceBefore=6,
            spaceAfter=6,
        ),
        "section_dark": ParagraphStyle(
            "PdfSectionDark",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "PdfBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "body_small": ParagraphStyle(
            "PdfBodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=INK,
            spaceAfter=4,
        ),
        "body_bold": ParagraphStyle(
            "PdfBodyBold",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceAfter=6,
        ),
        "sidebar_title": ParagraphStyle(
            "SidebarTitle",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.4,
            leading=10,
            textColor=MUTED,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "sidebar_meta": ParagraphStyle(
            "SidebarMeta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10,
            textColor=INK,
            spaceAfter=2,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.2,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=10.6,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=10,
            textColor=MUTED,
            spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.6,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def make_metric_card(title: str, metrics: list[tuple[str, str]], styles: dict[str, ParagraphStyle], width: float) -> Table:
    rows = [[Paragraph(title, styles["sidebar_title"])]]
    for label, value in metrics:
        rows.append(
            [
                Table(
                    [
                        [
                            Paragraph(html.escape(label), styles["metric_label"]),
                            Paragraph(html.escape(value), styles["metric_value"]),
                        ]
                    ],
                    colWidths=[width * 0.52, width * 0.33],
                    style=TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]
                    ),
                )
            ]
        )

    return Table(
        rows,
        colWidths=[width],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_FILL),
                ("BOX", (0, 0), (-1, -1), 0.8, LIGHT_BORDER),
                ("INNERGRID", (0, 1), (-1, -1), 0.5, LIGHT_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        ),
    )


def build_cover_story(
    report: dict[str, Any],
    chart_path: Path | None,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    request = report["request"]
    display_as_of_date = resolve_display_as_of_date(request)
    tables = {table["title"]: table for table in report.get("tables", [])}
    snapshot = rows_to_map(tables.get("Company Snapshot", {}).get("rows", []))
    performance = rows_to_map(tables.get("Market Performance", {}).get("rows", []))
    ratios_table = tables.get("Key Ratios", {})
    ratios = ratios_table.get("rows", [])[:4]

    investment_section = next(
        (section for section in report["sections"] if "Investment Recommendation" in section["section_name"]),
        None,
    )
    overview_section = next(
        (section for section in report["sections"] if section["section_name"] == "Company Overview"),
        None,
    )
    risk_section = next(
        (section for section in report["sections"] if section["section_name"] == "Risk Analysis"),
        None,
    )

    lead_paragraph = ""
    if investment_section and investment_section["paragraphs"]:
        lead_paragraph = investment_section["paragraphs"][0]["text"]
    elif overview_section and overview_section["paragraphs"]:
        lead_paragraph = overview_section["paragraphs"][0]["text"]

    overview_paragraphs: list[str] = []
    if overview_section:
        for para in overview_section["paragraphs"]:
            text = clean_text(para["text"])
            if "|" in text and "Ticker:" in text:
                continue
            overview_paragraphs.append(text)

    main_flowables: list[Any] = [
        Paragraph(request.get("company_name") or request["ticker"], styles["title"]),
        Paragraph(
            f"{request['ticker']} | As of {display_as_of_date} | "
            f"Verification: {str(report.get('overall_status', 'N/A')).upper()}",
            styles["subtitle"],
        ),
        Spacer(1, 6),
        Paragraph("Investment Overview", styles["section"]),
        Paragraph(markdown_to_paragraph_text(lead_paragraph), styles["body"]),
    ]

    if overview_paragraphs:
        main_flowables.append(Paragraph("Business Snapshot", styles["section"]))
        for para in overview_paragraphs[:1]:
            main_flowables.append(Paragraph(markdown_to_paragraph_text(para), styles["body"]))

    risk_sentences: list[str] = []
    if risk_section:
        for para in risk_section["paragraphs"]:
            risk_sentences.extend(split_sentences(para["text"]))
            if len(risk_sentences) >= 3:
                break
    if report.get("warnings"):
        risk_sentences = [str(item) for item in report["warnings"]] + risk_sentences
    risk_sentences = risk_sentences[:3]
    if risk_sentences:
        main_flowables.append(Paragraph("Key Risks", styles["section"]))
        for sentence in risk_sentences:
            main_flowables.append(Paragraph(f"• {html.escape(sentence)}", styles["body_small"]))

    left_block = KeepInFrame(MAIN_WIDTH, CONTENT_HEIGHT - 4, main_flowables, mode="shrink")

    sidebar_flowables: list[Any] = [
        Paragraph("Prepared by", styles["sidebar_title"]),
        Paragraph("ERRGen PDF Renderer", styles["sidebar_meta"]),
        Paragraph("Standalone export from existing run artifacts", styles["sidebar_meta"]),
        Spacer(1, 8),
        make_metric_card(
            "Key Report Data",
            [
                ("Ticker", request["ticker"]),
                ("Company", format_metric_value(request.get("company_name"))),
                ("Sector", format_metric_value(snapshot.get("Sector"))),
                ("Industry", format_metric_value(snapshot.get("Industry"))),
                ("Market Cap", format_metric_value(snapshot.get("Market Cap"))),
                ("Status", str(report.get("overall_status", "n/a")).upper()),
            ],
            styles,
            SIDEBAR_WIDTH - 8,
        ),
        Spacer(1, 8),
        make_metric_card(
            "Trading Snapshot",
            [
                ("Benchmark", format_metric_value(performance.get("Benchmark"))),
                ("Start Price", format_metric_value(performance.get("Company start price"))),
                ("End Price", format_metric_value(performance.get("Company end price"))),
                ("1Y Return", format_metric_value(performance.get("1-year company return"))),
                ("Excess Return", format_metric_value(performance.get("1-year excess return"))),
            ],
            styles,
            SIDEBAR_WIDTH - 8,
        ),
        Spacer(1, 8),
    ]

    if ratios:
        sidebar_flowables.append(
            make_metric_card(
                "Operating Metrics",
                [(row[0], row[1]) for row in ratios if len(row) >= 2],
                styles,
                SIDEBAR_WIDTH - 8,
            )
        )
        sidebar_flowables.append(Spacer(1, 8))

    if chart_path and chart_path.exists():
        sidebar_flowables.append(Paragraph("Performance Chart", styles["sidebar_title"]))
        sidebar_flowables.append(Image(str(chart_path), width=SIDEBAR_WIDTH - 10, height=1.62 * inch))

    right_block = KeepInFrame(SIDEBAR_WIDTH, CONTENT_HEIGHT - 4, sidebar_flowables, mode="shrink")

    return [left_block, FrameBreak(), right_block]


def build_table_flowable(table: dict[str, Any], styles: dict[str, ParagraphStyle], max_width: float) -> list[Any]:
    flowables: list[Any] = [
        Paragraph(table["title"], styles["section_dark"]),
    ]
    if table.get("description"):
        flowables.append(Paragraph(html.escape(table["description"]), styles["body_small"]))

    rows = table.get("rows", [])
    columns = table.get("columns", [])
    if not columns:
        return flowables

    wrapped_rows = [[Paragraph(html.escape(col), styles["body_small"]) for col in columns]]
    for row in rows:
        wrapped_rows.append(
            [Paragraph(html.escape(str(value)), styles["body_small"]) for value in row[: len(columns)]]
        )

    n_cols = max(len(columns), 1)
    base = max_width / n_cols
    col_widths = [base] * n_cols
    if n_cols == 2:
        col_widths = [max_width * 0.44, max_width * 0.56]
    elif n_cols >= 4:
        first_width = max_width * 0.24
        remainder = (max_width - first_width) / (n_cols - 1)
        col_widths = [first_width] + [remainder] * (n_cols - 1)

    table_flowable = LongTable(
        wrapped_rows,
        colWidths=col_widths,
        repeatRows=1,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8f8f8f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        ),
    )
    flowables.extend([Spacer(1, 4), table_flowable, Spacer(1, 10)])
    return flowables


def build_body_story(report: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    for section in report.get("sections", []):
        status = str(section.get("verification_status", "")).upper()
        title = f"{section['section_name']}  <font color='#666666' size='9'>{status}</font>"
        story.append(Paragraph(title, styles["section_dark"]))
        if section.get("unresolved_issues"):
            story.append(
                Paragraph(
                    f"{len(section['unresolved_issues'])} unresolved issue(s) remain in this section.",
                    styles["body_small"],
                )
            )
        for para in section.get("paragraphs", []):
            story.append(Paragraph(markdown_to_paragraph_text(para["text"]), styles["body"]))
        story.append(Spacer(1, 4))

    if report.get("tables"):
        story.append(PageBreak())
        story.append(Paragraph("Financial Summary", styles["section_dark"]))
        for table in report.get("tables", []):
            story.extend(build_table_flowable(table, styles, CONTENT_WIDTH))

    evidence_count = len(report.get("evidence_appendix", []))
    calc_count = len(report.get("calculation_appendix", []))
    story.append(Paragraph("Audit Summary", styles["section_dark"]))
    story.append(
        Paragraph(
            html.escape(
                f"Report ID: {report['report_id']} | "
                f"Evidence chunks cited: {evidence_count} | "
                f"Deterministic calculations: {calc_count}"
            ),
            styles["body_small"],
        )
    )
    story.append(
        Paragraph(
            "Full evidence, calculations, and revision artifacts remain available in the same run directory.",
            styles["body_small"],
        )
    )
    return story


def draw_cover_canvas(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 0.34 * inch, "US EQUITY RESEARCH")
    canvas.setFont("Helvetica", 10)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.34 * inch, "ERRGen Research")
    canvas.setStrokeColor(colors.HexColor("#2d2d2d"))
    canvas.setLineWidth(1)
    canvas.line(MARGIN, PAGE_HEIGHT - 0.46 * inch, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.46 * inch)

    canvas.setStrokeColor(LIGHT_BORDER)
    canvas.line(MARGIN, BOTTOM_MARGIN - 0.12 * inch, PAGE_WIDTH - MARGIN, BOTTOM_MARGIN - 0.12 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    disclaimer = (
        "Standalone PDF export generated from an existing ERRGen run. "
        "See run artifacts for full evidence and calculation traceability."
    )
    canvas.drawCentredString(PAGE_WIDTH / 2, BOTTOM_MARGIN - 0.28 * inch, disclaimer)
    canvas.restoreState()


def draw_body_canvas(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 10)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.35 * inch, "ERRGen Research")
    canvas.setStrokeColor(LIGHT_BORDER)
    canvas.line(MARGIN, PAGE_HEIGHT - 0.5 * inch, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 0.5 * inch)
    canvas.line(MARGIN, BOTTOM_MARGIN - 0.12 * inch, PAGE_WIDTH - MARGIN, BOTTOM_MARGIN - 0.12 * inch)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(MARGIN, BOTTOM_MARGIN - 0.28 * inch, "Evidence-grounded equity research export")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, BOTTOM_MARGIN - 0.28 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def render_pdf(run_dir: Path, output_path: Path) -> None:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json under {run_dir}")
    report = load_json(report_path)
    styles = build_styles()

    with tempfile.TemporaryDirectory(prefix="errgen_pdf_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        chart_path = build_price_chart(run_dir, tmp_dir)

        doc = BaseDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=f"{report['request'].get('company_name') or report['request']['ticker']} Research Report",
            author="ERRGen",
        )

        first_page_template = PageTemplate(
            id="cover",
            frames=[
                Frame(MARGIN, BOTTOM_MARGIN, MAIN_WIDTH, CONTENT_HEIGHT, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                Frame(MARGIN + MAIN_WIDTH + GUTTER, BOTTOM_MARGIN, SIDEBAR_WIDTH, CONTENT_HEIGHT, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            ],
            onPage=draw_cover_canvas,
        )
        body_template = PageTemplate(
            id="body",
            frames=[
                Frame(MARGIN, BOTTOM_MARGIN, CONTENT_WIDTH, CONTENT_HEIGHT, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            ],
            onPage=draw_body_canvas,
        )
        doc.addPageTemplates([first_page_template, body_template])

        story: list[Any] = []
        story.extend(build_cover_story(report, chart_path, styles))
        story.extend([NextPageTemplate("body"), PageBreak()])
        story.extend(build_body_story(report, styles))
        doc.build(story)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else run_dir / "report.pdf"
    render_pdf(run_dir, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
