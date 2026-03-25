"""
Offline tests for data-provider adapters.

These tests monkeypatch HTTP helpers and do not make network calls.
"""

from errgen.data.finnhub import FinnhubClient
from errgen.data.fmp import FMPClient
from errgen.data.sec import SECClient
from errgen.data.yahoo import YahooFinanceClient
from errgen.nodes.retrieve_data import _date_range, _lookback_window


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


def test_fmp_income_statement_filters_periods_after_as_of(monkeypatch):
    client = FMPClient(api_key="test-key")

    def fake_fmp_get(endpoint, extra_params=None):
        assert endpoint == "income-statement"
        return [
            {"period": "Q3", "calendarYear": "2025", "date": "2025-09-30", "revenue": 300},
            {"period": "Q2", "calendarYear": "2025", "date": "2025-06-30", "revenue": 200},
        ]

    monkeypatch.setattr(client, "_fmp_get", fake_fmp_get)

    source, chunks = client.get_income_statement(
        "NVDA",
        "quarter",
        as_of_date="2025-08-11",
    )

    assert source.api_source == "fmp"
    assert any(chunk.period == "Q2 2025" for chunk in chunks)
    assert not any(chunk.period == "Q3 2025" for chunk in chunks)


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


def test_yahoo_price_history_normalises_chart_payload(monkeypatch):
    client = YahooFinanceClient()

    def fake_get(url, params=None, headers=None):
        assert "/chart/NVDA" in url
        return {
            "chart": {
                "result": [
                    {
                        "timestamp": [1735689600, 1735776000],
                        "indicators": {
                            "quote": [
                                {
                                    "close": [100, 110],
                                    "high": [101, 111],
                                    "low": [99, 109],
                                    "open": [98, 105],
                                    "volume": [1000, 1200],
                                }
                            ]
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(client, "_get", fake_get)

    source, chunks = client.get_price_history("NVDA", "2025-01-01", "2025-01-02")

    assert source.api_source == "yahoo"
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


def test_sec_companyfacts_filters_future_periods(monkeypatch):
    client = SECClient(user_agent="ERRGen Test (contact: test@example.com)")

    monkeypatch.setattr(
        client,
        "_companyfacts",
        lambda ticker: (
            "0001234567",
            {
                "facts": {
                    "us-gaap": {
                        "RevenueFromContractWithCustomerExcludingAssessedTax": {
                            "units": {
                                "USD": [
                                    {
                                        "val": 200.0,
                                        "fy": 2025,
                                        "fp": "Q2",
                                        "end": "2025-06-30",
                                        "filed": "2025-08-01",
                                        "form": "10-Q",
                                    },
                                    {
                                        "val": 300.0,
                                        "fy": 2025,
                                        "fp": "Q3",
                                        "end": "2025-09-30",
                                        "filed": "2025-11-01",
                                        "form": "10-Q",
                                    },
                                ]
                            }
                        }
                    }
                }
            },
        ),
    )

    source, chunks = client.get_income_statement(
        "NVDA",
        "quarter",
        as_of_date="2025-08-11",
    )

    assert source.api_source == "sec_companyfacts"
    assert any(chunk.period == "Q2 2025" for chunk in chunks)
    assert not any(chunk.period == "Q3 2025" for chunk in chunks)


def test_retrieve_data_date_helpers_preserve_exact_dates():
    assert _date_range("2025-08-11") == ("2024-08-01", "2025-08-11")
    assert _lookback_window("2025-08-11", 30) == ("2025-07-12", "2025-08-11")
    assert _date_range("2025-02") == ("2024-02-01", "2025-02-28")
