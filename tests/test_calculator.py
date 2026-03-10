"""
Tests for the deterministic financial calculator.

These tests do NOT require any API keys and run fully offline.
They verify that every supported operation produces correct, reproducible results
and that edge cases (zero denominators, invalid operations) are handled cleanly.
"""

import math
import pytest

from errgen.calculator.finance_calc import FinanceCalculator, _safe_eval
from errgen.models import CalculationRequest


@pytest.fixture
def calc() -> FinanceCalculator:
    return FinanceCalculator()


# ---------------------------------------------------------------------------
# growth_rate / pct_change
# ---------------------------------------------------------------------------


def test_growth_rate_positive(calc):
    req = CalculationRequest(
        operation="growth_rate",
        inputs={"current": 60_922_000_000, "previous": 44_870_000_000},
        description="Revenue YoY",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - (60_922_000_000 - 44_870_000_000) / 44_870_000_000) < 1e-9


def test_growth_rate_negative(calc):
    req = CalculationRequest(
        operation="growth_rate",
        inputs={"current": 20_000, "previous": 25_000},
        description="Revenue decline",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - (-0.2)) < 1e-9


def test_growth_rate_zero_denominator(calc):
    req = CalculationRequest(
        operation="growth_rate",
        inputs={"current": 100, "previous": 0},
        description="Division by zero",
    )
    result = calc.compute(req)
    assert result.error is not None
    assert math.isnan(result.result)


def test_pct_change_alias(calc):
    req = CalculationRequest(
        operation="pct_change",
        inputs={"current": 110, "previous": 100},
        description="Alias test",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# margin
# ---------------------------------------------------------------------------


def test_gross_margin(calc):
    req = CalculationRequest(
        operation="margin",
        inputs={"numerator": 44_301_000_000, "denominator": 60_922_000_000},
        description="Gross margin",
    )
    result = calc.compute(req)
    assert result.error is None
    expected = 44_301_000_000 / 60_922_000_000
    assert abs(result.result - expected) < 1e-9


def test_margin_zero_denominator(calc):
    req = CalculationRequest(
        operation="margin",
        inputs={"numerator": 100, "denominator": 0},
        description="Zero denominator",
    )
    result = calc.compute(req)
    assert result.error is not None


# ---------------------------------------------------------------------------
# ratio
# ---------------------------------------------------------------------------


def test_ratio(calc):
    req = CalculationRequest(
        operation="ratio",
        inputs={"a": 150, "b": 50},
        description="3x ratio",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# current_ratio, debt_to_equity, net_debt
# ---------------------------------------------------------------------------


def test_current_ratio(calc):
    req = CalculationRequest(
        operation="current_ratio",
        inputs={"current_assets": 80_000, "current_liabilities": 40_000},
        description="Liquidity",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 2.0) < 1e-9


def test_debt_to_equity(calc):
    req = CalculationRequest(
        operation="debt_to_equity",
        inputs={"total_debt": 10_000, "equity": 50_000},
        description="Leverage",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 0.2) < 1e-9


def test_net_debt(calc):
    req = CalculationRequest(
        operation="net_debt",
        inputs={"total_debt": 30_000, "cash": 12_000},
        description="Net debt",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 18_000) < 1e-6


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------


def test_cagr(calc):
    req = CalculationRequest(
        operation="cagr",
        inputs={"start": 100, "end": 200, "years": 4},
        description="4-year CAGR",
    )
    result = calc.compute(req)
    assert result.error is None
    expected = (200 / 100) ** (1 / 4) - 1
    assert abs(result.result - expected) < 1e-9


def test_cagr_zero_start(calc):
    req = CalculationRequest(
        operation="cagr",
        inputs={"start": 0, "end": 200, "years": 4},
        description="Invalid CAGR",
    )
    result = calc.compute(req)
    assert result.error is not None


# ---------------------------------------------------------------------------
# YoY growth table
# ---------------------------------------------------------------------------


def test_yoy_growth_table(calc):
    series = [
        {"period": "FY2021", "value": 16_675_000_000},
        {"period": "FY2022", "value": 26_914_000_000},
        {"period": "FY2023", "value": 26_974_000_000},
        {"period": "FY2024", "value": 60_922_000_000},
    ]
    req = CalculationRequest(
        operation="yoy_growth_table",
        inputs={"series": series},
        description="Revenue YoY table",
    )
    result = calc.compute(req)
    assert result.error is None
    table = result.result["table"]
    assert len(table) == 4
    # Base period
    assert table[0]["yoy_growth"] is None
    # FY2022 vs FY2021
    expected_22 = (26_914_000_000 - 16_675_000_000) / 16_675_000_000
    assert abs(table[1]["yoy_growth"] - expected_22) < 1e-9
    # FY2024 should be high (>100%)
    assert table[3]["yoy_growth"] > 1.0


def test_yoy_growth_table_too_short(calc):
    req = CalculationRequest(
        operation="yoy_growth_table",
        inputs={"series": [{"period": "FY2024", "value": 100}]},
        description="Too short",
    )
    result = calc.compute(req)
    assert result.error is not None


# ---------------------------------------------------------------------------
# aggregate_sum
# ---------------------------------------------------------------------------


def test_aggregate_sum(calc):
    req = CalculationRequest(
        operation="aggregate_sum",
        inputs={"values": [100, 200, 300, 400]},
        description="Sum test",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 1000) < 1e-9


# ---------------------------------------------------------------------------
# arithmetic_expr (safe eval)
# ---------------------------------------------------------------------------


def test_safe_eval_basic():
    assert abs(_safe_eval("2 + 2") - 4) < 1e-9
    assert abs(_safe_eval("10 * (3 + 4)") - 70) < 1e-9
    assert abs(_safe_eval("100 / 4") - 25) < 1e-9
    assert abs(_safe_eval("2 ** 10") - 1024) < 1e-9


def test_safe_eval_rejects_names():
    with pytest.raises(ValueError):
        _safe_eval("import os")


def test_arithmetic_expr_via_calc(calc):
    req = CalculationRequest(
        operation="arithmetic_expr",
        inputs={"expression": "(60922 - 44870) / 44870"},
        description="Manual growth",
    )
    result = calc.compute(req)
    assert result.error is None
    expected = (60922 - 44870) / 44870
    assert abs(result.result - expected) < 1e-9


def test_arithmetic_expr_division_by_zero(calc):
    req = CalculationRequest(
        operation="arithmetic_expr",
        inputs={"expression": "100 / 0"},
        description="Zero div",
    )
    result = calc.compute(req)
    assert result.error is not None


# ---------------------------------------------------------------------------
# Unknown operation
# ---------------------------------------------------------------------------


def test_unknown_operation(calc):
    req = CalculationRequest(
        operation="definitely_not_a_real_operation",
        inputs={},
        description="Unknown",
    )
    result = calc.compute(req)
    assert result.error is not None


# ---------------------------------------------------------------------------
# Valuation ratios
# ---------------------------------------------------------------------------


def test_pe_ratio(calc):
    req = CalculationRequest(
        operation="pe_ratio",
        inputs={"price": 495.0, "eps": 11.93},
        description="P/E ratio",
    )
    result = calc.compute(req)
    assert result.error is None
    assert abs(result.result - 495.0 / 11.93) < 0.01


def test_ps_ratio(calc):
    req = CalculationRequest(
        operation="ps_ratio",
        inputs={"market_cap": 1_220_000_000_000, "revenue": 60_922_000_000},
        description="P/S ratio",
    )
    result = calc.compute(req)
    assert result.error is None


def test_ev_ebitda(calc):
    req = CalculationRequest(
        operation="ev_ebitda",
        inputs={"enterprise_value": 1_250_000_000_000, "ebitda": 37_000_000_000},
        description="EV/EBITDA",
    )
    result = calc.compute(req)
    assert result.error is None
