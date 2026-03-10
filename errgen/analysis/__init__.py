"""Analysis agents – one per major report section."""

from errgen.analysis.financial import FinancialAnalysisAgent
from errgen.analysis.news import NewsAnalysisAgent
from errgen.analysis.risk import RiskAnalysisAgent
from errgen.analysis.business import BusinessAnalysisAgent

__all__ = [
    "FinancialAnalysisAgent",
    "NewsAnalysisAgent",
    "RiskAnalysisAgent",
    "BusinessAnalysisAgent",
]
