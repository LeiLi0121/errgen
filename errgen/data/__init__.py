"""Data collection layer – provider adapters for real external APIs."""

from errgen.data.fmp import FMPClient
from errgen.data.newsapi import NewsAPIClient

__all__ = ["FMPClient", "NewsAPIClient"]
