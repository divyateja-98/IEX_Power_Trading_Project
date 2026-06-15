"""Run risk analytics for IEX electricity price forecasting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "risk"
ROLLING_WINDOW = 24
VAR_CONFIDENCE_LEVEL = 0.95


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the engineered dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Engineered dataset not found: {input_path}")

    logging.info("Loading engineered dataset: %s", input_path)
    dataframe = pd.read_csv(input_path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def resolve_target_column(dataframe: pd.DataFrame) -> str:
    """Resolve the MCP target column from possible dataset column names."""
    preferred_columns = [
        column
        for column in dataframe.columns
        if "MCP" in column and "Rs/MWh" in column and column.endswith("_market")
    ]
    if preferred_columns:
        return preferred_columns[0]

    fallback_columns = [
        column for column in dataframe.columns if "MCP" in column and "Rs/MWh" in column
    ]
    if fallback_columns:
        return fallback_columns[0]

    raise ValueError("No MCP target column found in the dataset.")


def prepare_risk_dataset(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Parse dates, sort observations, and create risk analytics fields."""
    required_columns = ["Date", "Hour", target_column]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}")

    risk_df = dataframe[required_columns].copy()
    risk_df["Date"] = pd.to_datetime(risk_df["Date"], errors="coerce")
    risk_df["Hour"] = pd.to_numeric(risk_df["Hour"], errors="coerce")
    risk_df[target_column] = pd.to_numeric(risk_df[target_column], errors="coerce")
    risk_df = risk_df.dropna(subset=["Date", "Hour", target_column])
    risk_df = risk_df.sort_values(["Date", "Hour"]).reset_index(drop=True)

    risk_df["timestamp"] = risk_df["Date"] + pd.to_timedelta(
        risk_df["Hour"].astype(int) - 1, unit="h"
    )
    risk_df["price_change"] = risk_df[target_column].diff()
    risk_df["price_return"] = risk_df[target_column].pct_change()
    risk_df["rolling_volatility"] = (
        risk_df["price_return"].rolling(window=ROLLING_WINDOW).std()
    )
    return risk_df


def calculate_risk_metrics(
    risk_df: pd.DataFrame, target_column: str
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate volatility, VaR, and price spike diagnostics."""
    price_returns = risk_df["price_return"].dropna()
    price_changes = risk_df["price_change"].dropna()
    prices = risk_df[target_column]

    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    spike_threshold = q3 + (1.5 * iqr)
    severe_spike_threshold = prices.quantile(0.95)

    risk_df = risk_df.copy()
    risk_df["price_spike_flag"] = (risk_df[target_column] > spike_threshold).astype(int)
    risk_df["severe_price_spike_flag"] = (
        risk_df[target_column] > severe_spike_threshold
    ).astype(int)

    value_at_risk = -price_returns.quantile(1 - VAR_CONFIDENCE_LEVEL)
    absolute_value_at_risk = -price_changes.quantile(1 - VAR_CONFIDENCE_LEVEL)

    metrics = {
        "price_volatility_std": float(prices.std()),
        "return_volatility_std": float(price_returns.std()),
        "rolling_volatility_mean": float(risk_df["rolling_volatility"].mean()),
        "rolling_volatility_max": float(risk_df["rolling_volatility"].max()),
        "var_95_return": float(value_at_risk),
        "var_95_price_change": float(absolute_value_at_risk),
        "spike_threshold_iqr": float(spike_threshold),
        "severe_spike_threshold_p95": float(severe_spike_threshold),
        "price_spike_count": int(risk_df["price_spike_flag"].sum()),
        "severe_price_spike_count": int(risk_df["severe_price_spike_flag"].sum()),
        "max_price": float(prices.max()),
        "mean_price": float(prices.mean()),
    }
    return metrics, risk_df


def save_figure(output_path: Path) -> None:
    """Save the active matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved chart: %s", output_path)


def plot_price_volatility(
    risk_df: pd.DataFrame, target_column: str, output_dir: Path
) -> Path:
    """Plot MCP over time to visualize price volatility."""
    output_path = output_dir / "price_volatility.png"
    plt.figure(figsize=(13, 6))
    sns.lineplot(data=risk_df, x="timestamp", y=target_column, linewidth=1)
    plt.title("MCP Price Volatility Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("MCP (Rs/MWh)")
    save_figure(output_path)
    return output_path


def plot_rolling_volatility(risk_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot rolling volatility."""
    output_path = output_dir / "rolling_volatility.png"
    plt.figure(figsize=(13, 6))
    sns.lineplot(data=risk_df, x="timestamp", y="rolling_volatility", color="#7a5c99")
    plt.title("24-Hour Rolling Return Volatility")
    plt.xlabel("Timestamp")
    plt.ylabel("Rolling Volatility")
    save_figure(output_path)
    return output_path


def plot_var_distribution(
    risk_df: pd.DataFrame, metrics: dict[str, float], output_dir: Path
) -> Path:
    """Plot return distribution and 95% VaR threshold."""
    output_path = output_dir / "var_95_distribution.png"
    returns = risk_df["price_return"].dropna()
    var_loss = metrics["var_95_return"]
    var_threshold = -var_loss

    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=80, kde=True, color="#2f6f73")
    plt.axvline(var_threshold, color="#b33b3b", linestyle="--", label="95% VaR")
    plt.title("Price Return Distribution with 95% VaR")
    plt.xlabel("Hourly Price Return")
    plt.ylabel("Frequency")
    plt.legend()
    save_figure(output_path)
    return output_path


def plot_price_spikes(
    risk_df: pd.DataFrame,
    target_column: str,
    metrics: dict[str, float],
    output_dir: Path,
) -> Path:
    """Plot detected price spikes over time."""
    output_path = output_dir / "price_spike_detection.png"
    spikes = risk_df[risk_df["price_spike_flag"] == 1]

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=risk_df, x="timestamp", y=target_column, linewidth=1, label="MCP")
    plt.scatter(
        spikes["timestamp"],
        spikes[target_column],
        color="#b33b3b",
        s=20,
        label="IQR price spike",
    )
    plt.axhline(
        metrics["spike_threshold_iqr"],
        color="#d28c45",
        linestyle="--",
        label="IQR spike threshold",
    )
    plt.title("Price Spike Detection")
    plt.xlabel("Timestamp")
    plt.ylabel("MCP (Rs/MWh)")
    plt.legend()
    save_figure(output_path)
    return output_path


def create_charts(
    risk_df: pd.DataFrame,
    target_column: str,
    metrics: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    """Create all risk analytics charts."""
    return [
        plot_price_volatility(risk_df, target_column, output_dir),
        plot_rolling_volatility(risk_df, output_dir),
        plot_var_distribution(risk_df, metrics, output_dir),
        plot_price_spikes(risk_df, target_column, metrics, output_dir),
    ]


def build_risk_report(
    risk_df: pd.DataFrame,
    target_column: str,
    metrics: dict[str, float],
    chart_paths: list[Path],
) -> str:
    """Build the risk analytics summary report."""
    top_spikes = risk_df.sort_values(target_column, ascending=False).head(10)
    top_spike_columns = ["timestamp", "Hour", target_column, "price_change", "price_return"]

    report_lines = [
        "IEX Power Price Risk Analytics Report",
        "=====================================",
        "",
        "Dataset:",
        f"- Rows analyzed: {len(risk_df)}",
        f"- Target column: {target_column}",
        f"- Rolling volatility window: {ROLLING_WINDOW} hours",
        "",
        "Price Volatility:",
        f"- MCP standard deviation: {metrics['price_volatility_std']:.6f}",
        f"- Hourly return volatility: {metrics['return_volatility_std']:.6f}",
        f"- Mean 24-hour rolling volatility: {metrics['rolling_volatility_mean']:.6f}",
        f"- Max 24-hour rolling volatility: {metrics['rolling_volatility_max']:.6f}",
        "",
        "95% Value at Risk:",
        f"- 95% return VaR: {metrics['var_95_return']:.6f}",
        f"- 95% absolute price-change VaR: {metrics['var_95_price_change']:.6f} Rs/MWh",
        "",
        "Price Spike Detection:",
        f"- IQR spike threshold: {metrics['spike_threshold_iqr']:.6f} Rs/MWh",
        f"- 95th percentile severe spike threshold: {metrics['severe_spike_threshold_p95']:.6f} Rs/MWh",
        f"- IQR price spike count: {metrics['price_spike_count']}",
        f"- Severe price spike count: {metrics['severe_price_spike_count']}",
        f"- Max MCP: {metrics['max_price']:.6f} Rs/MWh",
        f"- Mean MCP: {metrics['mean_price']:.6f} Rs/MWh",
        "",
        "Top 10 MCP spike observations:",
        top_spikes[top_spike_columns].to_string(index=False),
        "",
        "Charts generated:",
        "\n".join(f"- {path}" for path in chart_paths),
    ]
    return "\n".join(report_lines)


def save_outputs(
    risk_df: pd.DataFrame,
    report: str,
    output_dir: Path,
) -> None:
    """Save risk analytics dataset and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_dataset_path = output_dir / "risk_metrics_dataset.csv"
    report_path = output_dir / "risk_analytics_report.txt"

    risk_df.to_csv(risk_dataset_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    logging.info("Saved risk metrics dataset: %s", risk_dataset_path)
    logging.info("Saved risk analytics report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate risk analytics for IEX MCP price data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to engineered_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where risk outputs should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete risk analytics workflow."""
    configure_logging()
    args = parse_args()

    dataframe = load_dataset(args.input_path)
    target_column = resolve_target_column(dataframe)
    risk_df = prepare_risk_dataset(dataframe, target_column)
    metrics, risk_df = calculate_risk_metrics(risk_df, target_column)
    chart_paths = create_charts(risk_df, target_column, metrics, args.output_dir)
    report = build_risk_report(risk_df, target_column, metrics, chart_paths)

    print(report)
    save_outputs(risk_df, report, args.output_dir)


if __name__ == "__main__":
    main()
