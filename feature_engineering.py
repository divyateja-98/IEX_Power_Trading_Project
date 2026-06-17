"""Create forecasting features from the cleaned IEX dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from lineage.openlineage_config import lineage_run

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_dataset.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering_report.txt"
DEFAULT_LAGS = "1,2,24,48"
DEFAULT_ROLLING_WINDOWS = "24"


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


def validate_required_columns(dataframe: pd.DataFrame, target_column: str) -> None:
    """Validate columns needed for feature engineering."""
    required_columns = ["Date", "Hour", target_column]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}")


def prepare_time_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Parse date fields and sort observations in time order."""
    engineered = dataframe.copy()
    engineered["Date"] = pd.to_datetime(engineered["Date"], errors="coerce")
    engineered["Hour"] = pd.to_numeric(engineered["Hour"], errors="coerce")

    invalid_time_rows = int(engineered[["Date", "Hour"]].isna().any(axis=1).sum())
    if invalid_time_rows:
        raise ValueError(f"Found {invalid_time_rows} row(s) with invalid Date or Hour.")

    engineered = engineered.sort_values(["Date", "Hour"]).reset_index(drop=True)
    return engineered


def create_calendar_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features used by forecasting models."""
    engineered = dataframe.copy()
    engineered["hour"] = engineered["Hour"].astype(int)
    engineered["day"] = engineered["Date"].dt.day
    engineered["month"] = engineered["Date"].dt.month
    engineered["quarter"] = engineered["Date"].dt.quarter
    engineered["weekday"] = engineered["Date"].dt.weekday
    engineered["weekend_flag"] = engineered["weekday"].isin([5, 6]).astype(int)
    return engineered


def parse_int_csv(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("Expected one or more positive integers.")
    return parsed


def create_lag_features(
    dataframe: pd.DataFrame, target_column: str, lags: list[int]
) -> pd.DataFrame:
    """Create MCP lag features."""
    engineered = dataframe.copy()
    for lag in lags:
        engineered[f"lag_{lag}"] = engineered[target_column].shift(lag)
    return engineered


def create_rolling_features(
    dataframe: pd.DataFrame, target_column: str, rolling_windows: list[int]
) -> pd.DataFrame:
    """Create rolling MCP features from prior observations."""
    engineered = dataframe.copy()
    previous_target = engineered[target_column].shift(1)
    for window in rolling_windows:
        rolling = previous_target.rolling(window=window)
        engineered[f"rolling_mean_{window}"] = rolling.mean()
        engineered[f"rolling_std_{window}"] = rolling.std()
    return engineered


def engineer_features(
    dataframe: pd.DataFrame,
    target_column: str,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""
    engineered = prepare_time_index(dataframe)
    engineered[target_column] = pd.to_numeric(engineered[target_column], errors="coerce")
    engineered = create_calendar_features(engineered)
    engineered = create_lag_features(engineered, target_column, lags)
    engineered = create_rolling_features(engineered, target_column, rolling_windows)
    return engineered


def build_feature_report(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    target_column: str,
    engineered: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> str:
    """Create documentation for engineered features."""
    feature_docs = {
        "hour": "Hour of day copied from the source Hour column.",
        "day": "Day of month extracted from Date.",
        "month": "Month number extracted from Date.",
        "quarter": "Calendar quarter extracted from Date.",
        "weekday": "Day of week extracted from Date, where Monday is 0 and Sunday is 6.",
        "weekend_flag": "Binary flag equal to 1 for Saturday or Sunday, otherwise 0.",
    }
    for lag in lags:
        feature_docs[f"lag_{lag}"] = f"MCP value from {lag} observation(s) earlier."
    for window in rolling_windows:
        feature_docs[f"rolling_mean_{window}"] = (
            f"Mean MCP over the previous {window} observations, shifted to avoid leakage."
        )
        feature_docs[f"rolling_std_{window}"] = (
            f"Standard deviation of MCP over the previous {window} observations, shifted to avoid leakage."
        )
    engineered_features = list(feature_docs)
    missing_by_feature = engineered[engineered_features].isna().sum()

    report_lines = [
        "IEX Feature Engineering Report",
        "==============================",
        "",
        "Dataset:",
        f"- Input shape: {input_shape}",
        f"- Output shape: {output_shape}",
        f"- Target column: {target_column}",
        "",
        "Features created:",
    ]

    report_lines.extend(
        f"- {feature}: {description}"
        for feature, description in feature_docs.items()
    )
    report_lines.extend(
        [
            "",
            "Missing values introduced by lag and rolling features:",
            missing_by_feature.to_string(),
            "",
            "Notes:",
            "- Dataset was sorted by Date and Hour before lag and rolling features were created.",
            "- Rolling features use shifted MCP values so they only depend on historical observations.",
        ]
    )
    return "\n".join(report_lines)


def save_outputs(
    engineered: pd.DataFrame, report: str, output_path: Path, report_path: Path
) -> None:
    """Save engineered dataset and feature documentation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    engineered.to_csv(output_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    logging.info("Saved engineered dataset: %s", output_path)
    logging.info("Saved feature report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create forecasting features from cleaned IEX data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to cleaned_dataset.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where engineered_dataset.csv should be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where feature documentation should be saved.",
    )
    parser.add_argument(
        "--lags",
        default=DEFAULT_LAGS,
        help="Comma-separated MCP lag windows.",
    )
    parser.add_argument(
        "--rolling-windows",
        default=DEFAULT_ROLLING_WINDOWS,
        help="Comma-separated rolling windows for shifted MCP statistics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run feature engineering and save outputs."""
    configure_logging()
    args = parse_args()

    with lineage_run(
        "feature_engineering",
        inputs=[args.input_path],
        outputs=[args.output_path, args.report_path],
        metadata={
            "stage": "feature_engineering",
            "lags": args.lags,
            "rolling_windows": args.rolling_windows,
        },
    ):
        dataframe = load_dataset(args.input_path)
        target_column = resolve_target_column(dataframe)
        validate_required_columns(dataframe, target_column)
        lags = parse_int_csv(args.lags)
        rolling_windows = parse_int_csv(args.rolling_windows)

        engineered = engineer_features(dataframe, target_column, lags, rolling_windows)
        report = build_feature_report(
            input_shape=dataframe.shape,
            output_shape=engineered.shape,
            target_column=target_column,
            engineered=engineered,
            lags=lags,
            rolling_windows=rolling_windows,
        )
        print(report)

        save_outputs(engineered, report, args.output_path, args.report_path)


if __name__ == "__main__":
    main()
