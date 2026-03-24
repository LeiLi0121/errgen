"""Data collection layer – provider adapters for real external APIs."""

from errgen.data.finnhub import FinnhubClient
from errgen.data.fmp import FMPClient
from errgen.data.newsapi import NewsAPIClient
from errgen.data.sec import SECClient

__all__ = ["FMPClient", "FinnhubClient", "NewsAPIClient", "SECClient"]
