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
from datetime import date, datetime, time, timezone
from typing import Any

from errgen.config import Config
from errgen.data.base import BaseDataClient
from errgen.models import EvidenceChunk, SourceMetadata, SourceType

logger = logging.getLogger(__name__)


def _coerce_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


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

    def get_price_history(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        """
        Fetch daily price candles from Finnhub.

        Finnhub returns arrays keyed by field name. We normalise them into the
        same metadata shape used by FMP so downstream code can stay provider-agnostic.
        """
        start_dt = datetime.combine(date.fromisoformat(from_date), time.min, tzinfo=timezone.utc)
        end_dt = datetime.combine(date.fromisoformat(to_date), time.min, tzinfo=timezone.utc)
        start = int(start_dt.timestamp())
        end = int(end_dt.timestamp())
        raw = self._get(
            f"{self.base_url}/stock/candle",
            params={
                "symbol": ticker,
                "resolution": "D",
                "from": start,
                "to": end,
                "token": self.api_key,
            },
        )

        if not isinstance(raw, dict) or raw.get("s") != "ok":
            logger.warning("Finnhub: no price history for %s (%s – %s)", ticker, from_date, to_date)
            historical: list[dict[str, Any]] = []
        else:
            closes = raw.get("c") or []
            highs = raw.get("h") or []
            lows = raw.get("l") or []
            opens = raw.get("o") or []
            timestamps = raw.get("t") or []
            volumes = raw.get("v") or []
            count = min(len(closes), len(highs), len(lows), len(opens), len(timestamps), len(volumes))
            historical = []
            for idx in range(count):
                close = _coerce_float(closes[idx])
                high = _coerce_float(highs[idx])
                low = _coerce_float(lows[idx])
                open_ = _coerce_float(opens[idx])
                if close is None or high is None or low is None or open_ is None:
                    continue
                historical.append(
                    {
                        "date": datetime.utcfromtimestamp(int(timestamps[idx])).strftime("%Y-%m-%d"),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volumes[idx],
                    }
                )

        source = SourceMetadata(
            source_type=SourceType.PRICE_DATA,
            api_source="finnhub",
            document_identifier=f"finnhub_price_{ticker}_{from_date}_{to_date}",
            ticker=ticker,
            metadata={
                "endpoint": "/stock/candle",
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "historical": historical,
            },
        )

        chunks: list[EvidenceChunk] = []
        if historical:
            first = historical[0]
            last = historical[-1]
            highs = [float(item["high"]) for item in historical]
            lows = [float(item["low"]) for item in historical]
            summary = (
                f"Price history for {ticker} from {from_date} to {to_date}: "
                f"Opening price on {first['date']}: ${float(first['close']):.2f}, "
                f"Closing price on {last['date']}: ${float(last['close']):.2f}. "
                f"52-week high: ${max(highs):.2f}, "
                f"52-week low: ${min(lows):.2f}."
            )
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.PRICE_DATA,
                    text=summary,
                    metadata={
                        "ticker": ticker,
                        "from_date": from_date,
                        "to_date": to_date,
                        "data_points": len(historical),
                        "price_start": float(first["close"]),
                        "price_end": float(last["close"]),
                        "price_high": max(highs),
                        "price_low": min(lows),
                    },
                )
            )

        logger.info(
            "Finnhub: fetched price history for %s (%d data points)",
            ticker,
            len(historical),
        )
        return source, chunks
