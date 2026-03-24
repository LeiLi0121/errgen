"""
Offline tests for data-provider adapters.

These tests monkeypatch HTTP helpers and do not make network calls.
"""

from errgen.data.finnhub import FinnhubClient
from errgen.data.fmp import FMPClient
from errgen.data.sec import SECClient


def test_fmp_price_history_accepts_list_response(monkeypatch):
    client = FMPClient(api_key="test-key")

    def fake_fmp_get(endpoint, extra_params=None):
        assert endpoint == "historical-price-eod/full"
        return [
            {"date": "2025-01-02", "close": 110, "high": 111, "low": 109},
            {"date": "2025-01-01", "close": 100, "high": 101, "low": 99},
        ]

    monkeypatch.setattr(client, "_fmp_get", fake_fmp_get)

    source, chunks = client.get_price_history("NVDA", "2025-01-01", "2025-01-02")

    assert len(source.metadata["historical"]) == 2
    assert len(chunks) == 1
    assert chunks[0].metadata["price_start"] == 100.0
    assert chunks[0].metadata["price_end"] == 110.0


def test_finnhub_price_history_normalises_candles(monkeypatch):
    client = FinnhubClient(api_key="test-key")

    def fake_get(url, params=None, headers=None):
        assert url.endswith("/stock/candle")
        return {
            "s": "ok",
            "c": [100, 110],
            "h": [101, 111],
            "l": [99, 109],
            "o": [98, 105],
            "t": [1735689600, 1735776000],
            "v": [1000, 1200],
        }

    monkeypatch.setattr(client, "_get", fake_get)

    source, chunks = client.get_price_history("NVDA", "2025-01-01", "2025-01-02")

    assert source.api_source == "finnhub"
    assert len(source.metadata["historical"]) == 2
    assert chunks[0].metadata["price_start"] == 100.0
    assert chunks[0].metadata["price_end"] == 110.0


def test_sec_filings_use_official_submissions_and_archives(monkeypatch):
    client = SECClient(user_agent="ERRGen Test (contact: test@example.com)")

    monkeypatch.setattr(client, "_resolve_cik", lambda ticker: "0001234567")

    def fake_get(url, params=None, headers=None):
        assert url.endswith("/submissions/CIK0001234567.json")
        return {
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K"],
                    "filingDate": ["2025-05-01", "2025-05-03"],
                    "acceptanceDateTime": ["2025-05-01T16:30:00.000Z", "2025-05-03T12:00:00.000Z"],
                    "accessionNumber": ["0001234567-25-000010", "0001234567-25-000011"],
                    "primaryDocument": ["q1.htm", "event.htm"],
                    "primaryDocDescription": ["Quarterly report", "Current report"],
                    "reportDate": ["2025-04-30", "2025-05-03"],
                }
            }
        }

    def fake_get_text(url, params=None, headers=None):
        assert url.endswith("/1234567/000123456725000010/q1.htm")
        return """
        <html><body>
        Item 1. Business We design accelerated computing systems.
        Management's Discussion and Analysis Revenue increased meaningfully.
        Item 1A. Risk Factors Export controls and supply constraints remain relevant.
        </body></html>
        """

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr(client, "_get_text", fake_get_text)

    source, chunks = client.get_sec_filings(
        "NVDA",
        from_date="2025-01-01",
        to_date="2025-12-31",
        limit=2,
    )

    assert source.api_source == "sec"
    assert any(chunk.field_name == "filing_summary" for chunk in chunks)
    assert any(chunk.field_name == "filing_business" for chunk in chunks)
    assert any(chunk.field_name == "filing_risk_factors" for chunk in chunks)
