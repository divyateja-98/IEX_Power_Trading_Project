"""Generate Evidently AI monitoring reports for IEX power forecasting.

The script compares a historical reference window against a newer current
window from the project feature store. It produces:

- Data drift report for model input features.
- Target drift report for MCP.
- Prediction monitoring report using local model predictions.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "feature_store" / "historical_features.parquet"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "evidently"
TARGET_COLUMN = "mcp"
PREDICTION_COLUMN = "prediction"
METADATA_COLUMNS = {"timestamp", "date", "event_timestamp"}
FEATURE_COLUMNS = [
    "hour",
    "demand",
    "renewable_generation",
    "load_forecast",
    "rolling_mean_24",
    "rolling_std_24",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_48",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "solar_radiation",
    "weekday",
    "weekend_flag",
]


@dataclass(frozen=True)
class ReportPaths:
    """Output locations for Evidently reports."""

    data_drift: Path
    target_drift: Path
    prediction_monitoring: Path


def configure_logging() -> None:
    """Configure console logging for local and CI execution."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Evidently monitoring reports for IEX forecasting."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Feature store parquet or CSV used for monitoring.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local trained model artifact used to generate predictions.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory where Evidently reports are written.",
    )
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.7,
        help="Chronological fraction of rows used as the reference dataset.",
    )
    return parser.parse_args()


def import_evidently() -> tuple[Any, Any, Any, Any]:
    """Import Evidently classes lazily so the module can still be inspected."""
    try:
        from evidently.metric_preset import (
            DataDriftPreset,
            RegressionPreset,
            TargetDriftPreset,
        )
        from evidently.report import Report
    except ImportError as exc:
        raise SystemExit(
            "Evidently is not installed. Install project dependencies with "
            "`pip install -r requirements.txt` before running monitoring."
        ) from exc

    return Report, DataDriftPreset, TargetDriftPreset, RegressionPreset


def load_monitoring_data(input_path: Path) -> pd.DataFrame:
    """Load and normalize the monitoring dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Monitoring dataset not found: {input_path}")

    logging.info("Loading monitoring data from %s", input_path)
    if input_path.suffix.lower() == ".parquet":
        dataframe = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported monitoring data format: {input_path.suffix}")

    dataframe.columns = dataframe.columns.astype(str).str.strip()
    missing_columns = [
        column
        for column in [*FEATURE_COLUMNS, TARGET_COLUMN]
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required monitoring column(s): {missing_columns}")

    for column in ["event_timestamp", "timestamp", "date"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column], errors="coerce", utc=True
            )

    sort_columns = [
        column
        for column in ["event_timestamp", "timestamp", "date", "hour"]
        if column in dataframe.columns
    ]
    if sort_columns:
        dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)

    numeric_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    dataframe = dataframe.dropna(subset=numeric_columns).reset_index(drop=True)
    if dataframe.empty:
        raise ValueError("Monitoring dataset has no valid rows after cleaning.")

    logging.info("Loaded monitoring data with shape %s", dataframe.shape)
    return dataframe


def split_reference_current(
    dataframe: pd.DataFrame, reference_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into reference and current windows."""
    if not 0.1 <= reference_fraction <= 0.9:
        raise ValueError("reference_fraction must be between 0.1 and 0.9.")

    split_index = int(len(dataframe) * reference_fraction)
    if split_index <= 0 or split_index >= len(dataframe):
        raise ValueError("Reference/current split produced an empty dataset.")

    reference = dataframe.iloc[:split_index].copy()
    current = dataframe.iloc[split_index:].copy()
    logging.info(
        "Reference shape: %s, Current shape: %s", reference.shape, current.shape
    )
    return reference, current


def add_predictions(
    reference: pd.DataFrame, current: pd.DataFrame, model_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach model predictions required by prediction monitoring."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    logging.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    expected_features = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    expected_features = [str(feature) for feature in expected_features]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in reference.columns or feature not in current.columns
    ]
    if missing_features:
        raise ValueError(
            f"Model feature(s) missing from monitoring data: {missing_features}"
        )

    reference = reference.copy()
    current = current.copy()
    reference[PREDICTION_COLUMN] = model.predict(reference[expected_features])
    current[PREDICTION_COLUMN] = model.predict(current[expected_features])
    return reference, current


def report_paths(report_dir: Path) -> ReportPaths:
    """Create report directory and return report file paths."""
    report_dir.mkdir(parents=True, exist_ok=True)
    return ReportPaths(
        data_drift=report_dir / "data_drift_report.html",
        target_drift=report_dir / "target_drift_report.html",
        prediction_monitoring=report_dir / "prediction_monitoring_report.html",
    )


def build_report_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Limit report data to feature, target, and prediction columns."""
    columns = [*FEATURE_COLUMNS, TARGET_COLUMN, PREDICTION_COLUMN]
    return dataframe[columns].copy()


def generate_reports(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_paths: ReportPaths,
) -> None:
    """Generate Evidently data drift, target drift, and regression reports."""
    Report, DataDriftPreset, TargetDriftPreset, RegressionPreset = import_evidently()

    reference_report_df = build_report_frame(reference)
    current_report_df = build_report_frame(current)

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_report_df[FEATURE_COLUMNS],
        current_data=current_report_df[FEATURE_COLUMNS],
    )
    data_drift_report.save_html(str(output_paths.data_drift))

    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_report_df[[TARGET_COLUMN]],
        current_data=current_report_df[[TARGET_COLUMN]],
    )
    target_drift_report.save_html(str(output_paths.target_drift))

    prediction_report = Report(metrics=[RegressionPreset()])
    prediction_report.run(
        reference_data=reference_report_df,
        current_data=current_report_df,
        column_mapping={
            "target": TARGET_COLUMN,
            "prediction": PREDICTION_COLUMN,
            "numerical_features": FEATURE_COLUMNS,
        },
    )
    prediction_report.save_html(str(output_paths.prediction_monitoring))

    logging.info("Saved data drift report: %s", output_paths.data_drift)
    logging.info("Saved target drift report: %s", output_paths.target_drift)
    logging.info(
        "Saved prediction monitoring report: %s", output_paths.prediction_monitoring
    )


def main() -> None:
    """Run Evidently monitoring report generation."""
    configure_logging()
    args = parse_args()

    dataframe = load_monitoring_data(args.input_path)
    reference, current = split_reference_current(dataframe, args.reference_fraction)
    reference, current = add_predictions(reference, current, args.model_path)
    generate_reports(reference, current, report_paths(args.report_dir))


if __name__ == "__main__":
    main()
