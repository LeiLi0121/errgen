"""
NewsAPI data provider.

Endpoint used:
  /v2/everything – search by keyword within a date range

Registration: https://newsapi.org/
Free developer tier: 100 requests/day, articles up to 1 month old.
Upgrade to paid plan for production / longer history.

IMPORTANT: This client raises ValueError if NEWSAPI_KEY is missing.
"""

from __future__ import annotations

import logging
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


class NewsAPIClient(BaseDataClient):
    """
    Client for NewsAPI (/v2/everything endpoint).

    Returns (SourceMetadata, list[EvidenceChunk]) pairs with full article
    metadata so every news chunk carries its original URL and publication date.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or Config.NEWSAPI_KEY
        if not self.api_key:
            raise ValueError(
                "NEWSAPI_KEY is not set.\n"
                "Register at https://newsapi.org/ "
                "(free developer tier: 100 requests/day) "
                "and set NEWSAPI_KEY in .env"
            )
        self.base_url = Config.NEWSAPI_BASE_URL

    def search_news(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        language: str = "en",
        sort_by: str = "relevancy",
        page_size: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Search for news articles matching `query`.

        Dates should be ISO 8601 strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
        Returns up to `page_size` articles (default: Config.MAX_NEWS_ARTICLES).
        """
        page_size = page_size or Config.MAX_NEWS_ARTICLES

        params: dict[str, Any] = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),  # NewsAPI max per page
            "apiKey": self.api_key,
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        url = f"{self.base_url}/everything"
        raw = self._get(url, params=params)

        if raw.get("status") != "ok":
            raise ValueError(
                f"NewsAPI error: {raw.get('code', 'unknown')} – "
                f"{raw.get('message', 'No message')}"
            )

        articles: list[dict] = raw.get("articles", [])
        total_results = raw.get("totalResults", 0)

        source = SourceMetadata(
            source_type=SourceType.NEWS,
            api_source="newsapi",
            document_identifier=f"newsapi_{query[:40].replace(' ', '_')}",
            ticker=query,   # may not be a ticker; query stored for reference
            metadata={
                "endpoint": "/v2/everything",
                "query": query,
                "from_date": from_date,
                "to_date": to_date,
                "total_results": total_results,
            },
        )

        chunks: list[EvidenceChunk] = []
        for article in articles:
            title = article.get("title") or ""
            description = article.get("description") or ""
            content = article.get("content") or ""
            url_article = article.get("url") or ""
            published_at = article.get("publishedAt") or ""
            source_name = (article.get("source") or {}).get("name", "unknown")
            author = article.get("author") or ""

            # Skip removed / placeholder articles
            if title.lower().startswith("[removed]") or not title:
                continue

            body = content or description
            chunk_text = (
                f"[NEWS] {published_at} | {source_name}"
                + (f" | Author: {author}" if author else "")
                + f"\nTitle: {title}\nURL: {url_article}\n{body[:1500]}"
            )

            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.NEWS,
                    text=chunk_text,
                    metadata={
                        "title": title,
                        "published_at": published_at,
                        "source_name": source_name,
                        "url": url_article,
                        "author": author,
                        "query": query,
                    },
                )
            )

        logger.info(
            "NewsAPI: '%s' → %d articles returned (%d total matched)",
            query, len(chunks), total_results,
        )
        return source, chunks

    def get_company_news(
        self,
        ticker: str,
        company_name: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        page_size: int | None = None,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Convenience method that constructs a company-focused search query.
        Searches for `"TICKER" OR "Company Name"` to maximise coverage.
        """
        if company_name:
            query = f'"{ticker}" OR "{company_name}"'
        else:
            query = f'"{ticker}"'

        return self.search_news(
            query=query,
            from_date=from_date,
            to_date=to_date,
            sort_by="relevancy",
            page_size=page_size,
        )
