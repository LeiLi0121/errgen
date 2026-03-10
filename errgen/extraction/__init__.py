"""Information extraction layer – converts raw evidence chunks into structured facts."""

from errgen.extraction.financial import FinancialExtractor
from errgen.extraction.news import NewsExtractor

__all__ = ["FinancialExtractor", "NewsExtractor"]
