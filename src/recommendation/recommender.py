"""Generate procurement recommendations from MCP forecasts and risk outputs."""

from typing import Any


def recommend(
    current_mcp: float,
    forecast_mcp: float,
    volatility: float,
    buy_threshold_pct: float = 0.03,
    sell_threshold_pct: float = -0.03,
    high_volatility_threshold: float = 0.35,
) -> dict[str, Any]:
    """Return BUY NOW, WAIT, or SELL with an explanation."""
    if current_mcp <= 0:
        raise ValueError("Current MCP must be greater than zero.")
    if forecast_mcp <= 0:
        raise ValueError("Forecast MCP must be greater than zero.")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    expected_change = forecast_mcp - current_mcp
    expected_change_pct = expected_change / current_mcp

    if volatility >= high_volatility_threshold and abs(expected_change_pct) < 0.08:
        recommendation = "WAIT"
        explanation = (
            "Volatility is high and the forecast price move is not large enough to "
            "justify immediate action."
        )
    elif expected_change_pct >= buy_threshold_pct:
        recommendation = "BUY NOW"
        explanation = (
            "Forecast MCP is meaningfully above current MCP, so buying now can lock "
            "in a lower procurement price."
        )
    elif expected_change_pct <= sell_threshold_pct:
        recommendation = "SELL"
        explanation = (
            "Forecast MCP is meaningfully below current MCP, so selling or reducing "
            "exposure now can avoid the expected price decline."
        )
    else:
        recommendation = "WAIT"
        explanation = (
            "The price difference is within the neutral threshold, so waiting is preferred."
        )

    return {
        "recommendation": recommendation,
        "current_mcp": current_mcp,
        "forecast_mcp": forecast_mcp,
        "volatility": volatility,
        "expected_change": expected_change,
        "expected_change_pct": expected_change_pct,
        "explanation": explanation,
    }
