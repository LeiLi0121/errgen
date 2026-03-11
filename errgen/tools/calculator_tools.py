"""
Calculator tools for LangGraph agent tool calling.

These are @tool-decorated wrappers around the deterministic FinanceCalculator.
They can be used in two ways:

  1. Called directly inside any node function (as plain Python callables):

       result = calculate_growth_rate.invoke({"current": 100, "previous": 80})

  2. Bound to a LangChain ChatOpenAI model for LLM-driven tool calling:

       from langchain_openai import ChatOpenAI
       from errgen.tools import CALCULATOR_TOOLS

       llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(CALCULATOR_TOOLS)

     The LLM then calls the tools when it needs a number, and the result
     (including the calc_id) is returned so it can be cited in the paragraph.

Design principle: the LLM must NEVER compute financial arithmetic itself.
Any number that appears in a report paragraph should either come from
the pre-computed CalculationResult objects in the state or from a tool
call to this module.
"""

from __future__ import annotations

from langchain_core.tools import tool

from errgen.calculator.finance_calc import FinanceCalculator
from errgen.models import CalculationRequest

_calc = FinanceCalculator()


def _run(operation: str, inputs: dict, description: str) -> str:
    """Execute a calculation and return a human-readable result string."""
    req = CalculationRequest(
        operation=operation,
        inputs=inputs,
        description=description,
    )
    res = _calc.compute(req)
    if res.error:
        return f"ERROR [{operation}]: {res.error}"
    result_str = (
        f"{res.result:.4f}" if isinstance(res.result, float) else str(res.result)
    )
    return (
        f"calc_id={res.calc_id} | "
        f"operation={res.operation} | "
        f"result={result_str} | "
        f"formula={res.formula_description}"
    )


@tool
def calculate_growth_rate(
    current: float,
    previous: float,
    description: str = "",
) -> str:
    """
    Calculate period-over-period growth rate: (current - previous) / abs(previous).

    Use for year-over-year or quarter-over-quarter revenue, income, or any
    metric growth.  Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "growth_rate",
        {"current": current, "previous": previous},
        description or f"Growth rate: {previous} → {current}",
    )


@tool
def calculate_margin(
    numerator: float,
    denominator: float,
    description: str = "",
) -> str:
    """
    Calculate a margin or percentage ratio: numerator / denominator.

    Use for gross margin, operating margin, net margin, FCF margin, etc.
    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "margin",
        {"numerator": numerator, "denominator": denominator},
        description or f"Margin: {numerator} / {denominator}",
    )


@tool
def calculate_cagr(
    start_value: float,
    end_value: float,
    years: float,
    description: str = "",
) -> str:
    """
    Calculate Compound Annual Growth Rate: (end / start)^(1 / years) - 1.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "cagr",
        {"start_value": start_value, "end_value": end_value, "years": years},
        description or f"CAGR over {years} years",
    )


@tool
def calculate_current_ratio(
    current_assets: float,
    current_liabilities: float,
    description: str = "",
) -> str:
    """
    Calculate the liquidity current ratio: current_assets / current_liabilities.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "current_ratio",
        {
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
        },
        description or "Current ratio",
    )


@tool
def calculate_debt_to_equity(
    total_debt: float,
    equity: float,
    description: str = "",
) -> str:
    """
    Calculate the leverage debt-to-equity ratio: total_debt / equity.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "debt_to_equity",
        {"total_debt": total_debt, "equity": equity},
        description or "Debt-to-equity ratio",
    )


@tool
def calculate_net_debt(
    total_debt: float,
    cash: float,
    description: str = "",
) -> str:
    """
    Calculate net debt: total_debt - cash.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "net_debt",
        {"total_debt": total_debt, "cash": cash},
        description or "Net debt",
    )


@tool
def calculate_pe_ratio(
    price: float,
    eps: float,
    description: str = "",
) -> str:
    """
    Calculate the Price-to-Earnings (P/E) valuation ratio: price / EPS.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "pe_ratio",
        {"price": price, "eps": eps},
        description or "P/E ratio",
    )


@tool
def calculate_ps_ratio(
    market_cap: float,
    revenue: float,
    description: str = "",
) -> str:
    """
    Calculate the Price-to-Sales (P/S) ratio: market_cap / revenue.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "ps_ratio",
        {"market_cap": market_cap, "revenue": revenue},
        description or "P/S ratio",
    )


@tool
def calculate_ev_ebitda(
    enterprise_value: float,
    ebitda: float,
    description: str = "",
) -> str:
    """
    Calculate the EV/EBITDA valuation multiple: enterprise_value / EBITDA.

    Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "ev_ebitda",
        {"enterprise_value": enterprise_value, "ebitda": ebitda},
        description or "EV/EBITDA multiple",
    )


@tool
def evaluate_expression(expression: str, description: str = "") -> str:
    """
    Safely evaluate an arithmetic expression string, e.g. '(100 - 80) / 80'.

    Only numeric literals and the operators +, -, *, /, (, ), ** are allowed.
    No variables.  Use this for ad-hoc calculations not covered by the
    other tools.  Returns the calc_id so it can be cited in a paragraph.
    """
    return _run(
        "arithmetic_expr",
        {"expression": expression},
        description or f"Expression: {expression}",
    )


# ---------------------------------------------------------------------------
# Convenience export for LLM tool binding
# ---------------------------------------------------------------------------

CALCULATOR_TOOLS = [
    calculate_growth_rate,
    calculate_margin,
    calculate_cagr,
    calculate_current_ratio,
    calculate_debt_to_equity,
    calculate_net_debt,
    calculate_pe_ratio,
    calculate_ps_ratio,
    calculate_ev_ebitda,
    evaluate_expression,
]
