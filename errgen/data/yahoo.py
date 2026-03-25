"""
Yahoo Finance data provider.

Uses the unofficial chart endpoint for daily OHLCV history:
  /v8/finance/chart/{ticker}

No API key is required. This provider is used only as a fallback for price
history when paid financial APIs are unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

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


class YahooFinanceClient(BaseDataClient):
    """Client for Yahoo Finance chart data."""

    def __init__(self) -> None:
        self.base_url = "https://query1.finance.yahoo.com/v8/finance"

    def get_price_history(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> tuple[SourceMetadata, list[EvidenceChunk]]:
        raw = self._get(
            f"{self.base_url}/chart/{ticker}",
            params={
                "interval": "1d",
                "period1": int(datetime.fromisoformat(from_date).timestamp()),
                "period2": int(datetime.fromisoformat(to_date).timestamp()),
                "includeAdjustedClose": "true",
            },
        )

        result = ((raw or {}).get("chart") or {}).get("result") or []
        payload = result[0] if result else {}
        timestamps = payload.get("timestamp") or []
        quote = (((payload.get("indicators") or {}).get("quote") or [{}])[0]) or {}
        closes = quote.get("close") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        opens = quote.get("open") or []
        volumes = quote.get("volume") or []

        historical: list[dict[str, Any]] = []
        count = min(len(timestamps), len(closes), len(highs), len(lows), len(opens), len(volumes))
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
            api_source="yahoo",
            document_identifier=f"yahoo_price_{ticker}_{from_date}_{to_date}",
            ticker=ticker,
            metadata={
                "endpoint": "/v8/finance/chart",
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
            chunks.append(
                EvidenceChunk(
                    source_id=source.source_id,
                    source_type=SourceType.PRICE_DATA,
                    text=(
                        f"Price history for {ticker} from {from_date} to {to_date}: "
                        f"Opening price on {first['date']}: ${float(first['close']):.2f}, "
                        f"Closing price on {last['date']}: ${float(last['close']):.2f}. "
                        f"52-week high: ${max(highs):.2f}, "
                        f"52-week low: ${min(lows):.2f}."
                    ),
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
            "Yahoo: fetched price history for %s (%d data points)",
            ticker,
            len(historical),
        )
        return source, chunks
