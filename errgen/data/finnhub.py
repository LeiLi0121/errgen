"""
Finnhub data provider (company news).

Endpoint: GET /v1/company-news
  - symbol: ticker (e.g. AAPL)
  - from: YYYY-MM-DD
  - to: YYYY-MM-DD
  - token: API key

Registration: https://finnhub.io/
Free tier: 60 API calls/minute. Company news is included.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


class FinnhubClient(BaseDataClient):
    """
    Client for Finnhub company-news API.

    Returns (SourceMetadata, list[EvidenceChunk]) with the same shape as
    NewsAPIClient / FMP get_stock_news for use in the pipeline.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or Config.FINNHUB_API_KEY
        self.base_url = Config.FINNHUB_BASE_URL.rstrip("/")

    def get_company_news(
        self,
        ticker: str,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch company news for the given ticker and date range.

        from_date / to_date should be YYYY-MM-DD. Returns up to `limit` articles
        (default: Config.MAX_NEWS_ARTICLES).
        """
        limit = limit or Config.MAX_NEWS_ARTICLES
        params: dict[str, Any] = {
            "symbol": ticker,
            "token": self.api_key,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        url = f"{self.base_url}/company-news"
        raw = self._get(url, params=params)

        if not raw or not isinstance(raw, list):
            logger.warning("Finnhub: no news array for %s", ticker)
            raw = []

        source = SourceMetadata(
            source_type=SourceType.NEWS,
            api_source="finnhub",
            document_identifier=f"finnhub_news_{ticker}",
            ticker=ticker,
            metadata={
                "endpoint": "/company-news",
                "ticker": ticker,
                "from_date": from_date,
                "to_date": to_date,
            },
        )

        chunks: list[EvidenceChunk] = []
        for i, article in enumerate(raw[:limit]):
            if not isinstance(article, dict):
                continue
            # Finnhub returns: headline, summary, datetime (unix), source, url, category, etc.
            headline = article.get("headline") or article.get("title") or ""
            summary = article.get("summary") or article.get("text") or ""
            url_article = article.get("url") or ""
            source_name = article.get("source", "unknown")
            ts = article.get("datetime")
            if ts is not None:
                try:
                    pub_at = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
                except (TypeError, OSError):
                    pub_at = str(ts)
            else:
                pub_at = ""

            if not headline and not summary:
                continue

            body = summary or headline
            chunk_text = (
                f"[NEWS] {pub_at} | {source_name}\n"
                f"Title: {headline}\n"
                f"URL: {url_article}\n"
                f"{body[:1500]}"
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.NEWS,
                    text=chunk_text,
                    metadata={
                        "title": headline,
                        "published_at": pub_at,
                        "source_name": source_name,
                        "url": url_article,
                        "ticker": ticker,
                    },
                )
            )

        logger.info(
            "Finnhub: fetched %d news articles for %s",
            len(chunks),
            ticker,
        )
        return source, chunks
