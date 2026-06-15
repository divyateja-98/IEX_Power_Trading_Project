"""Run exploratory data analysis for electricity price forecasting."""

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
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "eda_summary_report.txt"


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}")

    logging.info("Loading cleaned dataset: %s", input_path)
    dataframe = pd.read_csv(input_path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def resolve_target_column(dataframe: pd.DataFrame) -> str:
    """Resolve the MCP target column from possible merged dataset names."""
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


def prepare_eda_features(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Add parsed date and aggregated weather features for EDA."""
    eda_df = dataframe.copy()
    eda_df["Date"] = pd.to_datetime(eda_df["Date"], errors="coerce")
    eda_df["Month"] = eda_df["Date"].dt.to_period("M").dt.to_timestamp()

    temperature_columns = [
        column for column in eda_df.columns if "temperature_2m" in column
    ]
    wind_columns = [column for column in eda_df.columns if "wind_speed_10m" in column]
    radiation_columns = [
        column for column in eda_df.columns if "shortwave_radiation" in column
    ]

    eda_df["avg_temperature"] = eda_df[temperature_columns].mean(axis=1)
    eda_df["avg_wind_speed"] = eda_df[wind_columns].mean(axis=1)
    eda_df["avg_solar_radiation"] = eda_df[radiation_columns].mean(axis=1)
    eda_df[target_column] = pd.to_numeric(eda_df[target_column], errors="coerce")

    return eda_df


def save_figure(output_path: Path) -> None:
    """Save the active matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved chart: %s", output_path)


def plot_histogram(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> Path:
    """Create MCP histogram."""
    output_path = output_dir / "mcp_histogram.png"
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[target_column], bins=50, kde=True, color="#2f6f73")
    plt.title("MCP Distribution")
    plt.xlabel("MCP (Rs/MWh)")
    plt.ylabel("Frequency")
    save_figure(output_path)
    return output_path


def plot_boxplot(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> Path:
    """Create MCP boxplot."""
    output_path = output_dir / "mcp_boxplot.png"
    plt.figure(figsize=(9, 5))
    sns.boxplot(x=dataframe[target_column], color="#d28c45")
    plt.title("MCP Boxplot")
    plt.xlabel("MCP (Rs/MWh)")
    save_figure(output_path)
    return output_path


def plot_correlation_heatmap(
    dataframe: pd.DataFrame, target_column: str, output_dir: Path
) -> Path:
    """Create a compact correlation heatmap around key forecasting variables."""
    output_path = output_dir / "correlation_heatmap.png"
    candidate_columns = [
        target_column,
        "Hour",
        "Purchase Bid (MW)_market",
        "Sell Bid (MW)_market",
        "MCV (MW)_market",
        "Final Scheduled Volume (MW)_market",
        "avg_temperature",
        "avg_wind_speed",
        "avg_solar_radiation",
    ]
    selected_columns = [column for column in candidate_columns if column in dataframe.columns]
    correlation = dataframe[selected_columns].corr(numeric_only=True)

    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    save_figure(output_path)
    return output_path


def plot_hourly_trend(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> Path:
    """Create hourly MCP trend."""
    output_path = output_dir / "hourly_mcp_trend.png"
    hourly = dataframe.groupby("Hour", as_index=False)[target_column].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hourly, x="Hour", y=target_column, marker="o", color="#4f6fbd")
    plt.title("Hourly MCP Trend")
    plt.xlabel("Hour")
    plt.ylabel("Average MCP (Rs/MWh)")
    save_figure(output_path)
    return output_path


def plot_monthly_trend(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> Path:
    """Create monthly MCP trend."""
    output_path = output_dir / "monthly_mcp_trend.png"
    monthly = dataframe.dropna(subset=["Month"]).groupby("Month", as_index=False)[
        target_column
    ].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="Month", y=target_column, marker="o", color="#7a5c99")
    plt.title("Monthly MCP Trend")
    plt.xlabel("Month")
    plt.ylabel("Average MCP (Rs/MWh)")
    plt.xticks(rotation=45)
    save_figure(output_path)
    return output_path


def plot_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    target_column: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> Path:
    """Create a sampled scatter plot for a weather variable against MCP."""
    sample = dataframe[[x_column, target_column]].dropna()
    if len(sample) > 6000:
        sample = sample.sample(n=6000, random_state=42)

    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=sample,
        x=x_column,
        y=target_column,
        scatter_kws={"alpha": 0.25, "s": 14},
        line_kws={"color": "#b33b3b"},
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("MCP (Rs/MWh)")
    save_figure(output_path)
    return output_path


def create_charts(dataframe: pd.DataFrame, target_column: str, output_dir: Path) -> list[Path]:
    """Create all requested EDA charts."""
    chart_paths = [
        plot_histogram(dataframe, target_column, output_dir),
        plot_boxplot(dataframe, target_column, output_dir),
        plot_correlation_heatmap(dataframe, target_column, output_dir),
        plot_hourly_trend(dataframe, target_column, output_dir),
        plot_monthly_trend(dataframe, target_column, output_dir),
        plot_scatter(
            dataframe,
            "avg_temperature",
            target_column,
            "Temperature vs MCP",
            "Average Temperature",
            output_dir / "temperature_vs_mcp.png",
        ),
        plot_scatter(
            dataframe,
            "avg_wind_speed",
            target_column,
            "Wind Speed vs MCP",
            "Average Wind Speed",
            output_dir / "wind_speed_vs_mcp.png",
        ),
        plot_scatter(
            dataframe,
            "avg_solar_radiation",
            target_column,
            "Solar Radiation vs MCP",
            "Average Solar Radiation",
            output_dir / "solar_radiation_vs_mcp.png",
        ),
    ]
    return chart_paths


def build_summary_report(
    dataframe: pd.DataFrame, target_column: str, chart_paths: list[Path]
) -> str:
    """Build a text EDA summary report."""
    numeric_columns = dataframe.select_dtypes(include="number").columns
    target_stats = dataframe[target_column].describe()
    correlations = (
        dataframe[numeric_columns]
        .corr(numeric_only=True)[target_column]
        .drop(labels=[target_column], errors="ignore")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(10)
    )
    hourly = dataframe.groupby("Hour")[target_column].mean()
    monthly = dataframe.dropna(subset=["Month"]).groupby("Month")[target_column].mean()

    report_lines = [
        "IEX Electricity Price Forecasting EDA Summary",
        "============================================",
        "",
        "Dataset:",
        f"- Rows: {dataframe.shape[0]}",
        f"- Columns: {dataframe.shape[1]}",
        f"- Target column used: {target_column}",
        "",
        "Target summary:",
        target_stats.to_string(),
        "",
        "Trend highlights:",
        f"- Highest average hourly MCP: Hour {hourly.idxmax()} ({hourly.max():.2f})",
        f"- Lowest average hourly MCP: Hour {hourly.idxmin()} ({hourly.min():.2f})",
    ]

    if not monthly.empty:
        report_lines.extend(
            [
                f"- Highest average monthly MCP: {monthly.idxmax().date()} ({monthly.max():.2f})",
                f"- Lowest average monthly MCP: {monthly.idxmin().date()} ({monthly.min():.2f})",
            ]
        )

    report_lines.extend(
        [
            "",
            "Top absolute correlations with MCP:",
            correlations.to_string(),
            "",
            "Weather feature correlations:",
            f"- Average temperature vs MCP: {dataframe['avg_temperature'].corr(dataframe[target_column]):.4f}",
            f"- Average wind speed vs MCP: {dataframe['avg_wind_speed'].corr(dataframe[target_column]):.4f}",
            f"- Average solar radiation vs MCP: {dataframe['avg_solar_radiation'].corr(dataframe[target_column]):.4f}",
            "",
            "Charts generated:",
            "\n".join(f"- {path}" for path in chart_paths),
        ]
    )
    return "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate EDA charts and summary for electricity price forecasting."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to cleaned_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where EDA charts should be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the EDA summary report should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete EDA workflow."""
    configure_logging()
    args = parse_args()

    dataframe = load_dataset(args.input_path)
    target_column = resolve_target_column(dataframe)
    dataframe = prepare_eda_features(dataframe, target_column)

    chart_paths = create_charts(dataframe, target_column, args.output_dir)
    report = build_summary_report(dataframe, target_column, chart_paths)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")

    print(report)
    logging.info("Saved EDA summary report: %s", args.report_path)


if __name__ == "__main__":
    main()
