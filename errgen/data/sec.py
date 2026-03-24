"""
SEC EDGAR data provider.

Uses official SEC endpoints for:
  - ticker -> CIK lookup
  - company submissions metadata
  - filing document retrieval from EDGAR archives
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    no_script = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    no_tags = re.sub(r"(?s)<[^>]+>", " ", no_script)
    return _normalise_whitespace(html.unescape(no_tags))


class SECClient(BaseDataClient):
    """Client for SEC EDGAR submissions and filing documents."""

    _ticker_cache: dict[str, str] | None = None

    def __init__(self, user_agent: str | None = None) -> None:
        self.base_url = Config.SEC_BASE_URL.rstrip("/")
        self.archives_base_url = Config.SEC_ARCHIVES_BASE_URL.rstrip("/")
        self.tickers_url = Config.SEC_TICKERS_URL
        self.user_agent = user_agent or Config.SEC_USER_AGENT

    def _sec_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }

    def _resolve_cik(self, ticker: str) -> str:
        if SECClient._ticker_cache is None:
            raw = self._get(self.tickers_url, headers=self._sec_headers())
            mapping: dict[str, str] = {}
            if isinstance(raw, dict):
                for item in raw.values():
                    if not isinstance(item, dict):
                        continue
                    symbol = str(item.get("ticker", "")).upper()
                    cik = item.get("cik_str")
                    if symbol and cik is not None:
                        mapping[symbol] = str(int(cik)).zfill(10)
            SECClient._ticker_cache = mapping

        cik = SECClient._ticker_cache.get(ticker.upper()) if SECClient._ticker_cache else None
        if not cik:
            raise ValueError(f"SEC: no CIK mapping found for ticker '{ticker}'")
        return cik

    @staticmethod
    def _extract_filing_sections(text: str) -> list[tuple[str, str]]:
        section_specs = [
            ("filing_business", [r"item\s*1\.?\s*business", r"business overview"]),
            ("filing_mda", [r"management['’]s discussion and analysis", r"item\s*2\.?\s*management"]),
            ("filing_risk_factors", [r"item\s*1a\.?\s*risk factors", r"risk factors"]),
        ]
        lower = text.lower()
        snippets: list[tuple[str, str]] = []
        for field_name, patterns in section_specs:
            start = -1
            for pattern in patterns:
                match = re.search(pattern, lower, flags=re.IGNORECASE)
                if match:
                    start = match.start()
                    break
            if start < 0:
                continue
            excerpt = text[start:start + 2400].strip()
            if excerpt:
                snippets.append((field_name, excerpt))

        if snippets:
            return snippets

        fallback = text[:2400].strip()
        return [("filing_excerpt", fallback)] if fallback else []

    @staticmethod
    def _recent_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
        recent = submissions.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []

        def _value(field: str, idx: int, total: int) -> Any:
            values = recent.get(field) or [None] * total
            if idx >= len(values):
                return None
            return values[idx]

        forms = recent.get("form") or []
        total = len(forms)
        records: list[dict[str, Any]] = []
        for idx in range(total):
            records.append(
                {
                    "form": forms[idx],
                    "filingDate": _value("filingDate", idx, total),
                    "acceptanceDateTime": _value("acceptanceDateTime", idx, total),
                    "accessionNumber": _value("accessionNumber", idx, total),
                    "primaryDocument": _value("primaryDocument", idx, total),
                    "primaryDocDescription": _value("primaryDocDescription", idx, total),
                    "reportDate": _value("reportDate", idx, total),
                }
            )
        return records

    def get_sec_filings(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch recent 10-K / 10-Q filings from SEC EDGAR and extract summary chunks.
        """
        limit = limit or Config.MAX_SEC_FILINGS
        cik = self._resolve_cik(ticker)
        submissions_url = f"{self.base_url}/submissions/CIK{cik}.json"
        submissions = self._get(submissions_url, headers=self._sec_headers())

        filings = []
        for item in self._recent_filings(submissions):
            form_type = (item.get("form") or "").upper()
            filed_at = item.get("filingDate") or ""
            if form_type not in {"10-K", "10-Q"}:
                continue
            if filed_at and (filed_at < from_date or filed_at > to_date):
                continue
            filings.append(item)
            if len(filings) >= limit:
                break

        source = SourceMetadata(
            source_type=SourceType.FILING,
            api_source="sec",
            document_identifier=f"sec_filings_{ticker}_{from_date}_{to_date}",
            url=submissions_url,
            ticker=ticker,
            metadata={
                "endpoint": "submissions/CIK.json",
                "ticker": ticker,
                "cik": cik,
                "from": from_date,
                "to": to_date,
                "n_filings": len(filings),
            },
        )

        chunks: list[EvidenceChunk] = []
        cik_numeric = str(int(cik))
        for filing in filings:
            form_type = (filing.get("form") or "").upper()
            filed_at = filing.get("filingDate") or ""
            accepted_at = filing.get("acceptanceDateTime") or ""
            accession = filing.get("accessionNumber") or ""
            primary_document = filing.get("primaryDocument") or ""
            filing_title = filing.get("primaryDocDescription") or f"{ticker} {form_type} filing"
            accession_compact = accession.replace("-", "")
            final_link = ""
            if accession_compact and primary_document:
                final_link = (
                    f"{self.archives_base_url}/{cik_numeric}/"
                    f"{accession_compact}/{primary_document}"
                )

            metadata = {
                "ticker": ticker,
                "cik": cik,
                "form_type": form_type,
                "filed_at": filed_at,
                "accepted_at": accepted_at,
                "accession_number": accession,
                "url": final_link,
                "filing_title": filing_title,
            }
            summary_text = (
                f"[FILING] {ticker} {form_type} | Filed: {filed_at} | Accepted: {accepted_at}\n"
                f"Title: {filing_title}\n"
                f"URL: {final_link or 'N/A'}"
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.FILING,
                    text=summary_text,
                    field_name="filing_summary",
                    period=filed_at[:10] if filed_at else None,
                    metadata=metadata,
                )
            )

            if not final_link:
                continue

            try:
                raw_text = self._get_text(final_link, headers=self._sec_headers())
                cleaned = _strip_html(raw_text)
            except Exception as exc:
                logger.warning("SEC: failed to fetch filing body for %s %s: %s", ticker, form_type, exc)
                continue

            if not cleaned:
                continue

            for section_name, excerpt in self._extract_filing_sections(cleaned):
                chunks.append(
                    EvidenceChunk(
                        source_id=source.source_id,
                        source_type=SourceType.FILING,
                        text=excerpt,
                        field_name=section_name,
                        period=filed_at[:10] if filed_at else None,
                        metadata=metadata,
                    )
                )

        logger.info("SEC: fetched %d filing chunks for %s", len(chunks), ticker)
        return source, chunks
