"""
Deterministic financial calculator.

This module deliberately avoids LLM arithmetic.  All computations are pure
Python, fully deterministic, and unit-tested.  Analysis agents submit
CalculationRequest objects; the calculator returns CalculationResult objects
with full formula documentation so the checker can verify numeric claims.

Supported operations
--------------------
  growth_rate       – YoY / QoQ percentage change
  margin            – a / b (e.g. gross profit / revenue)
  ratio             – generic a / b
  pct_change        – synonym for growth_rate
  safe_divide       – a / b guarded against zero denominator
  yoy_growth_table  – list of (period, value, yoy_growth) for a metric series
  cagr              – compound annual growth rate
  current_ratio     – current assets / current liabilities
  debt_to_equity    – total debt / stockholders' equity
  net_debt          – total debt - cash
  ebitda_margin     – ebitda / revenue
  fcf_margin        – free_cash_flow / revenue
  rd_intensity      – r&d / revenue
  pe_ratio          – price / eps
  ps_ratio          – market_cap / revenue
  ev_ebitda         – enterprise_value / ebitda
  aggregate_sum     – sum of a list of values
  arithmetic_expr   – evaluate a safe arithmetic expression string
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any

from errgen.models import CalculationRequest, CalculationResult

logger = logging.getLogger(__name__)

# Allowed operators for safe expression evaluation
_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """
    Evaluate a simple arithmetic expression safely (no exec, no eval).
    Only supports +, -, *, /, ** and numeric literals.
    Raises ValueError for unsupported constructs.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expr!r}") from exc

    def _eval(node: ast.expr) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {op_type}")
            left = _eval(node.left)
            right = _eval(node.right)
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero in expression")
            return _SAFE_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return _SAFE_OPS[op_type](_eval(node.operand))
        raise ValueError(f"Unsupported AST node: {type(node)}")

    return _eval(tree)


class FinanceCalculator:
    """
    Stateless financial calculator.

    Usage:
        calc = FinanceCalculator()
        req = CalculationRequest(
            operation="growth_rate",
            inputs={"current": 60_922_000_000, "previous": 44_870_000_000},
            description="NVIDIA FY2024 vs FY2023 revenue YoY growth",
            chunk_ids_used=["chunk_abc", "chunk_def"],
        )
        result = calc.compute(req)
        # result.result == 0.3578...
    """

    def compute(self, request: CalculationRequest) -> CalculationResult:
        """Dispatch to the appropriate calculation method and return a result."""
        op = request.operation.lower()
        try:
            result, formula = self._dispatch(op, request.inputs)
            return CalculationResult(
                calc_id=request.calc_id,
                operation=op,
                inputs=request.inputs,
                result=result,
                formula_description=formula,
                description=request.description,
                error=None,
            )
        except (ZeroDivisionError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Calculator error for operation=%s: %s", op, exc
            )
            return CalculationResult(
                calc_id=request.calc_id,
                operation=op,
                inputs=request.inputs,
                result=float("nan"),
                formula_description=f"Error: {exc}",
                description=request.description,
                error=str(exc),
            )

    def compute_many(
        self, requests: list[CalculationRequest]
    ) -> list[CalculationResult]:
        return [self.compute(r) for r in requests]

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    def _dispatch(
        self, op: str, inputs: dict[str, Any]
    ) -> tuple[float | dict, str]:
        ops = {
            "growth_rate": self._growth_rate,
            "pct_change": self._growth_rate,
            "margin": self._margin,
            "ratio": self._ratio,
            "safe_divide": self._ratio,
            "yoy_growth_table": self._yoy_growth_table,
            "cagr": self._cagr,
            "current_ratio": self._current_ratio,
            "debt_to_equity": self._debt_to_equity,
            "net_debt": self._net_debt,
            "ebitda_margin": self._ebitda_margin,
            "fcf_margin": self._fcf_margin,
            "rd_intensity": self._rd_intensity,
            "pe_ratio": self._pe_ratio,
            "ps_ratio": self._ps_ratio,
            "ev_ebitda": self._ev_ebitda,
            "aggregate_sum": self._aggregate_sum,
            "arithmetic_expr": self._arithmetic_expr,
        }
        if op not in ops:
            raise ValueError(
                f"Unknown operation '{op}'. "
                f"Supported: {sorted(ops)}"
            )
        return ops[op](inputs)

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    @staticmethod
    def _growth_rate(inputs: dict) -> tuple[float, str]:
        current = float(inputs["current"])
        previous = float(inputs["previous"])
        if previous == 0:
            raise ZeroDivisionError("Previous value is zero; cannot compute growth rate.")
        result = (current - previous) / abs(previous)
        formula = f"({current} - {previous}) / |{previous}| = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _margin(inputs: dict) -> tuple[float, str]:
        numerator = float(inputs["numerator"])
        denominator = float(inputs["denominator"])
        if denominator == 0:
            raise ZeroDivisionError("Denominator is zero; cannot compute margin.")
        result = numerator / denominator
        formula = f"{numerator} / {denominator} = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _ratio(inputs: dict) -> tuple[float, str]:
        a = float(inputs["a"])
        b = float(inputs["b"])
        if b == 0:
            raise ZeroDivisionError("Denominator b is zero; cannot compute ratio.")
        result = a / b
        formula = f"{a} / {b} = {result:.4f}"
        return result, formula

    @staticmethod
    def _yoy_growth_table(inputs: dict) -> tuple[dict, str]:
        """
        Compute YoY growth for a time series.
        inputs: {"series": [{"period": "FY2021", "value": 1000}, ...]}
        Ordered chronologically (oldest first).
        """
        series: list[dict] = inputs["series"]
        if len(series) < 2:
            raise ValueError("Need at least 2 data points for YoY growth table.")

        table = []
        for i, item in enumerate(series):
            row: dict[str, Any] = {"period": item["period"], "value": item["value"]}
            if i > 0:
                prev_val = float(series[i - 1]["value"])
                curr_val = float(item["value"])
                if prev_val == 0:
                    row["yoy_growth"] = None
                    row["yoy_growth_pct"] = "N/A"
                else:
                    growth = (curr_val - prev_val) / abs(prev_val)
                    row["yoy_growth"] = growth
                    row["yoy_growth_pct"] = f"{growth:.2%}"
            else:
                row["yoy_growth"] = None
                row["yoy_growth_pct"] = "N/A (base period)"
            table.append(row)

        formula = "YoY growth = (current - previous) / |previous| for each period"
        return {"table": table}, formula

    @staticmethod
    def _cagr(inputs: dict) -> tuple[float, str]:
        start = float(inputs["start"])
        end = float(inputs["end"])
        years = float(inputs["years"])
        if start <= 0:
            raise ValueError("Start value must be positive for CAGR.")
        if years <= 0:
            raise ValueError("Years must be positive for CAGR.")
        result = (end / start) ** (1.0 / years) - 1.0
        formula = f"({end} / {start}) ^ (1 / {years}) - 1 = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _current_ratio(inputs: dict) -> tuple[float, str]:
        ca = float(inputs["current_assets"])
        cl = float(inputs["current_liabilities"])
        if cl == 0:
            raise ZeroDivisionError("Current liabilities is zero.")
        result = ca / cl
        formula = f"current_assets ({ca}) / current_liabilities ({cl}) = {result:.2f}"
        return result, formula

    @staticmethod
    def _debt_to_equity(inputs: dict) -> tuple[float, str]:
        debt = float(inputs["total_debt"])
        equity = float(inputs["equity"])
        if equity == 0:
            raise ZeroDivisionError("Equity is zero.")
        result = debt / equity
        formula = f"total_debt ({debt}) / equity ({equity}) = {result:.2f}"
        return result, formula

    @staticmethod
    def _net_debt(inputs: dict) -> tuple[float, str]:
        debt = float(inputs["total_debt"])
        cash = float(inputs["cash"])
        result = debt - cash
        formula = f"total_debt ({debt}) - cash ({cash}) = {result}"
        return result, formula

    @staticmethod
    def _ebitda_margin(inputs: dict) -> tuple[float, str]:
        ebitda = float(inputs["ebitda"])
        revenue = float(inputs["revenue"])
        if revenue == 0:
            raise ZeroDivisionError("Revenue is zero.")
        result = ebitda / revenue
        formula = f"EBITDA ({ebitda}) / revenue ({revenue}) = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _fcf_margin(inputs: dict) -> tuple[float, str]:
        fcf = float(inputs["free_cash_flow"])
        revenue = float(inputs["revenue"])
        if revenue == 0:
            raise ZeroDivisionError("Revenue is zero.")
        result = fcf / revenue
        formula = f"FCF ({fcf}) / revenue ({revenue}) = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _rd_intensity(inputs: dict) -> tuple[float, str]:
        rd = float(inputs["rd_expenses"])
        revenue = float(inputs["revenue"])
        if revenue == 0:
            raise ZeroDivisionError("Revenue is zero.")
        result = rd / revenue
        formula = f"R&D ({rd}) / revenue ({revenue}) = {result:.4f} ({result:.2%})"
        return result, formula

    @staticmethod
    def _pe_ratio(inputs: dict) -> tuple[float, str]:
        price = float(inputs["price"])
        eps = float(inputs["eps"])
        if eps == 0:
            raise ZeroDivisionError("EPS is zero.")
        result = price / eps
        formula = f"price ({price}) / EPS ({eps}) = {result:.2f}x"
        return result, formula

    @staticmethod
    def _ps_ratio(inputs: dict) -> tuple[float, str]:
        mkt_cap = float(inputs["market_cap"])
        revenue = float(inputs["revenue"])
        if revenue == 0:
            raise ZeroDivisionError("Revenue is zero.")
        result = mkt_cap / revenue
        formula = f"market_cap ({mkt_cap}) / revenue ({revenue}) = {result:.2f}x"
        return result, formula

    @staticmethod
    def _ev_ebitda(inputs: dict) -> tuple[float, str]:
        ev = float(inputs["enterprise_value"])
        ebitda = float(inputs["ebitda"])
        if ebitda == 0:
            raise ZeroDivisionError("EBITDA is zero.")
        result = ev / ebitda
        formula = f"EV ({ev}) / EBITDA ({ebitda}) = {result:.2f}x"
        return result, formula

    @staticmethod
    def _aggregate_sum(inputs: dict) -> tuple[float, str]:
        values: list = inputs["values"]
        total = sum(float(v) for v in values)
        formula = f"sum({values}) = {total}"
        return total, formula

    @staticmethod
    def _arithmetic_expr(inputs: dict) -> tuple[float, str]:
        expr: str = inputs["expression"]
        result = _safe_eval(expr)
        formula = f"evaluate({expr!r}) = {result}"
        return result, formula


# ---------------------------------------------------------------------------
# Convenience factory functions for common calculations
# ---------------------------------------------------------------------------


def build_growth_request(
    description: str,
    current: float,
    previous: float,
    chunk_ids: list[str] | None = None,
) -> CalculationRequest:
    return CalculationRequest(
        operation="growth_rate",
        inputs={"current": current, "previous": previous},
        description=description,
        chunk_ids_used=chunk_ids or [],
    )


def build_margin_request(
    description: str,
    numerator: float,
    denominator: float,
    chunk_ids: list[str] | None = None,
) -> CalculationRequest:
    return CalculationRequest(
        operation="margin",
        inputs={"numerator": numerator, "denominator": denominator},
        description=description,
        chunk_ids_used=chunk_ids or [],
    )


def build_yoy_table_request(
    description: str,
    series: list[dict],
    chunk_ids: list[str] | None = None,
) -> CalculationRequest:
    return CalculationRequest(
        operation="yoy_growth_table",
        inputs={"series": series},
        description=description,
        chunk_ids_used=chunk_ids or [],
    )
