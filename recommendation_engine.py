"""Generate procurement recommendations from MCP forecast and volatility."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "recommendation_report.txt"


@dataclass(frozen=True)
class RecommendationResult:
    """Structured procurement recommendation output."""

    recommendation: str
    current_mcp: float
    forecast_mcp: float
    volatility: float
    expected_change: float
    expected_change_pct: float
    explanation: str


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def validate_inputs(current_mcp: float, forecast_mcp: float, volatility: float) -> None:
    """Validate recommendation inputs."""
    if current_mcp <= 0:
        raise ValueError("Current MCP must be greater than zero.")
    if forecast_mcp <= 0:
        raise ValueError("Forecast MCP must be greater than zero.")
    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")


def generate_recommendation(
    current_mcp: float,
    forecast_mcp: float,
    volatility: float,
    buy_threshold_pct: float = 0.03,
    sell_threshold_pct: float = -0.03,
    high_volatility_threshold: float = 0.35,
) -> RecommendationResult:
    """Generate BUY NOW, WAIT, or SELL recommendation."""
    validate_inputs(current_mcp, forecast_mcp, volatility)

    expected_change = forecast_mcp - current_mcp
    expected_change_pct = expected_change / current_mcp

    if volatility >= high_volatility_threshold and abs(expected_change_pct) < 0.08:
        recommendation = "WAIT"
        explanation = (
            "Volatility is high and the forecast price move is not large enough to "
            "justify immediate procurement or selling action. Waiting reduces the "
            "risk of acting on a noisy short-term price signal."
        )
    elif expected_change_pct >= buy_threshold_pct:
        recommendation = "BUY NOW"
        explanation = (
            "Forecast MCP is meaningfully above current MCP, so buying now can lock "
            "in a lower procurement price before the expected price increase."
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
            "The difference between current MCP and forecast MCP is within the neutral "
            "threshold, so there is no strong price signal. Waiting is preferred until "
            "the forecast direction becomes clearer."
        )

    return RecommendationResult(
        recommendation=recommendation,
        current_mcp=current_mcp,
        forecast_mcp=forecast_mcp,
        volatility=volatility,
        expected_change=expected_change,
        expected_change_pct=expected_change_pct,
        explanation=explanation,
    )


def build_report(result: RecommendationResult) -> str:
    """Build a text report for the recommendation."""
    return "\n".join(
        [
            "IEX Procurement Recommendation Report",
            "====================================",
            "",
            "Inputs:",
            f"- Current MCP: {result.current_mcp:.6f} Rs/MWh",
            f"- Forecast MCP: {result.forecast_mcp:.6f} Rs/MWh",
            f"- Volatility: {result.volatility:.6f}",
            "",
            "Price Signal:",
            f"- Expected MCP change: {result.expected_change:.6f} Rs/MWh",
            f"- Expected MCP change percentage: {result.expected_change_pct:.6%}",
            "",
            "Recommendation:",
            f"- {result.recommendation}",
            "",
            "Explanation:",
            result.explanation,
        ]
    )


def save_report(report: str, report_path: Path) -> None:
    """Save recommendation report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logging.info("Saved recommendation report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate BUY NOW, WAIT, or SELL recommendation for MCP procurement."
    )
    parser.add_argument("--current-mcp", type=float, required=True, help="Current MCP.")
    parser.add_argument("--forecast-mcp", type=float, required=True, help="Forecast MCP.")
    parser.add_argument(
        "--volatility",
        type=float,
        required=True,
        help="Current or rolling volatility value.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the recommendation report should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the recommendation engine."""
    configure_logging()
    args = parse_args()

    result = generate_recommendation(
        current_mcp=args.current_mcp,
        forecast_mcp=args.forecast_mcp,
        volatility=args.volatility,
    )
    report = build_report(result)
    print(report)
    save_report(report, args.report_path)


if __name__ == "__main__":
    main()
