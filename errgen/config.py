"""
Central configuration for ERRGen.

All configuration is read from environment variables (loaded from .env via
python-dotenv).  The Config.validate_required() method is called at pipeline
startup to fail fast with clear instructions if keys are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Config:
    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    # Optional: override the API base URL (e.g. for local vLLM: http://127.0.0.1:8000/v1)
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "")
    # Primary model – used for analysis, verification, revision
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
    # Fast model – used for extraction tasks where cost matters more
    OPENAI_FAST_MODEL: str = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    # Maximum tokens for a single LLM response
    OPENAI_MAX_TOKENS: int = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))
    # Temperature for generation (lower = more deterministic)
    OPENAI_TEMPERATURE: float = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))

    # ------------------------------------------------------------------
    # Financial Modeling Prep (primary financial data)
    # Register free at https://financialmodelingprep.com/developer/docs/
    # Free tier: 250 requests/day, covers financials + news
    # ------------------------------------------------------------------
    FMP_API_KEY: str = os.environ.get("FMP_API_KEY", "")
    FMP_BASE_URL: str = "https://financialmodelingprep.com/stable"

    # ------------------------------------------------------------------
    # NewsAPI (supplemental news retrieval)
    # Register free at https://newsapi.org/ – 100 req/day on free tier
    # ------------------------------------------------------------------
    NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")
    NEWSAPI_BASE_URL: str = "https://newsapi.org/v2"

    # ------------------------------------------------------------------
    # Finnhub (company news, optional)
    # Register free at https://finnhub.io/ – 60 calls/min on free tier
    # ------------------------------------------------------------------
    FINNHUB_API_KEY: str = os.environ.get("FINNHUB_API_KEY", "")
    FINNHUB_BASE_URL: str = os.environ.get("FINNHUB_BASE_URL", "https://finnhub.io/api/v1")

    # ------------------------------------------------------------------
    # SEC EDGAR (official filings source, no API key required)
    # SEC expects a descriptive User-Agent with contact info.
    # ------------------------------------------------------------------
    SEC_BASE_URL: str = os.environ.get("SEC_BASE_URL", "https://data.sec.gov")
    SEC_ARCHIVES_BASE_URL: str = os.environ.get(
        "SEC_ARCHIVES_BASE_URL",
        "https://www.sec.gov/Archives/edgar/data",
    )
    SEC_TICKERS_URL: str = os.environ.get(
        "SEC_TICKERS_URL",
        "https://www.sec.gov/files/company_tickers.json",
    )
    SEC_USER_AGENT: str = os.environ.get(
        "SEC_USER_AGENT",
        "ERRGen/1.0 research pipeline (contact: leili0121@outlook.com)",
    )

    # ------------------------------------------------------------------
    # Pipeline tuning
    # ------------------------------------------------------------------
    # Maximum checker→reviser iterations before marking as unresolved
    MAX_REVISION_ITERATIONS: int = int(os.environ.get("MAX_REVISION_ITERATIONS", "3"))
    # How many news articles to pull per query
    MAX_NEWS_ARTICLES: int = int(os.environ.get("MAX_NEWS_ARTICLES", "20"))
    # How many financial periods (quarters or annual) to retrieve
    MAX_FINANCIAL_PERIODS: int = int(os.environ.get("MAX_FINANCIAL_PERIODS", "4"))
    # How many SEC filings to retrieve and process
    MAX_SEC_FILINGS: int = int(os.environ.get("MAX_SEC_FILINGS", "3"))
    # Benchmark ticker used for relative market-performance analysis
    BENCHMARK_TICKER: str = os.environ.get("BENCHMARK_TICKER", "SPY")
    # Price-history windows
    PRICE_LOOKBACK_DAYS: int = int(os.environ.get("PRICE_LOOKBACK_DAYS", "365"))
    PREDICTION_LOOKBACK_DAYS: int = int(os.environ.get("PREDICTION_LOOKBACK_DAYS", "30"))
    # Number of LLM call retries on transient failures
    LLM_RETRY_ATTEMPTS: int = int(os.environ.get("LLM_RETRY_ATTEMPTS", "3"))
    # Seconds between retries
    LLM_RETRY_DELAY: float = float(os.environ.get("LLM_RETRY_DELAY", "5.0"))
    # Concurrent checker requests per verification_agent round
    CHECKER_MAX_CONCURRENCY: int = int(os.environ.get("CHECKER_MAX_CONCURRENCY", "4"))
    # Concurrent revision requests per revise_sections round
    REVISION_MAX_CONCURRENCY: int = int(os.environ.get("REVISION_MAX_CONCURRENCY", "4"))
    # Concurrent section-level pairwise judge requests per sample
    EVAL_PAIRWISE_MAX_CONCURRENCY: int = int(
        os.environ.get("EVAL_PAIRWISE_MAX_CONCURRENCY", "6")
    )

    # ------------------------------------------------------------------
    # Run artifacts
    # ------------------------------------------------------------------
    RUNS_DIR: str = os.environ.get("RUNS_DIR", "runs")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_required(cls) -> None:
        """
        Raise a clear ValueError if any required API key is missing.
        Called once at pipeline startup so users know exactly what to configure.
        """
        missing: list[str] = []
        instructions: dict[str, str] = {}

        if not cls.OPENAI_API_KEY and not cls.OPENAI_BASE_URL:
            missing.append("OPENAI_API_KEY")
            instructions["OPENAI_API_KEY"] = (
                "Register at https://platform.openai.com/ and create an API key. "
                "Or set OPENAI_BASE_URL to point to a local vLLM / compatible server."
            )
        if not cls.FMP_API_KEY:
            missing.append("FMP_API_KEY")
            instructions["FMP_API_KEY"] = (
                "Register at https://financialmodelingprep.com/developer/docs/ "
                "(free tier covers income statement, balance sheet, cash flow, news)."
            )
        if not cls.NEWSAPI_KEY:
            missing.append("NEWSAPI_KEY")
            instructions["NEWSAPI_KEY"] = (
                "Register at https://newsapi.org/ "
                "(free developer tier: 100 requests/day)."
            )

        if missing:
            lines = [
                "ERRGen: Missing required API keys. Refusing to start.",
                "Copy .env.example to .env and fill in the following:",
                "",
            ]
            for key in missing:
                lines.append(f"  {key}")
                lines.append(f"    → {instructions[key]}")
                lines.append("")
            raise ValueError("\n".join(lines))

    @classmethod
    def as_dict(cls) -> dict:
        """Return non-secret configuration for logging."""
        return {
            "openai_model": cls.OPENAI_MODEL,
            "openai_fast_model": cls.OPENAI_FAST_MODEL,
            "fmp_base_url": cls.FMP_BASE_URL,
            "newsapi_base_url": cls.NEWSAPI_BASE_URL,
            "finnhub_base_url": cls.FINNHUB_BASE_URL,
            "sec_base_url": cls.SEC_BASE_URL,
            "max_revision_iterations": cls.MAX_REVISION_ITERATIONS,
            "max_news_articles": cls.MAX_NEWS_ARTICLES,
            "max_financial_periods": cls.MAX_FINANCIAL_PERIODS,
            "max_sec_filings": cls.MAX_SEC_FILINGS,
            "benchmark_ticker": cls.BENCHMARK_TICKER,
            "price_lookback_days": cls.PRICE_LOOKBACK_DAYS,
            "prediction_lookback_days": cls.PREDICTION_LOOKBACK_DAYS,
            "checker_max_concurrency": cls.CHECKER_MAX_CONCURRENCY,
            "revision_max_concurrency": cls.REVISION_MAX_CONCURRENCY,
            "eval_pairwise_max_concurrency": cls.EVAL_PAIRWISE_MAX_CONCURRENCY,
            "runs_dir": cls.RUNS_DIR,
        }
