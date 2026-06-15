"""Prepare a Feast-compatible IEX power trading feature source."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered.csv"
DEFAULT_PARQUET_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered.parquet"

CANONICAL_FEATURE_COLUMNS = [
    "timestamp",
    "date",
    "hour",
    "event_timestamp",
    "mcp",
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


def configure_logging() -> None:
    """Configure console logging for feature store preparation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def normalize_column_name(column: str) -> str:
    """Normalize a column name for source column matching."""
    return column.lower().replace("_", " ").replace("-", " ")


def load_engineered_dataset(input_path: Path) -> pd.DataFrame:
    """Load the engineered forecasting dataset."""
    if not input_path.exists():
        raise FileNotFoundError(f"Engineered dataset not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def find_columns(
    dataframe: pd.DataFrame,
    required_tokens: list[str],
    excluded_tokens: list[str] | None = None,
) -> list[str]:
    """Find columns whose normalized names contain every required token."""
    required = [token.lower() for token in required_tokens]
    excluded = [token.lower() for token in excluded_tokens or []]
    matches = []
    for column in dataframe.columns:
        normalized = normalize_column_name(column)
        if all(token in normalized for token in required) and not any(
            token in normalized for token in excluded
        ):
            matches.append(column)
    return matches


def first_matching_column(
    dataframe: pd.DataFrame,
    required_tokens: list[str],
    excluded_tokens: list[str] | None = None,
) -> str:
    """Return the first matching source column or raise a helpful error."""
    matches = find_columns(dataframe, required_tokens, excluded_tokens)
    if not matches:
        tokens = ", ".join(required_tokens)
        raise ValueError(f"No source column found for tokens: {tokens}")
    return matches[0]


def resolve_target_column(dataframe: pd.DataFrame) -> str:
    """Resolve the MCP target column used for model labels."""
    market_mcp_columns = [
        column
        for column in find_columns(dataframe, ["mcp"])
        if "market" in normalize_column_name(column)
    ]
    if market_mcp_columns:
        return market_mcp_columns[0]
    return first_matching_column(dataframe, ["mcp"])


def resolve_demand_column(dataframe: pd.DataFrame) -> str:
    """Resolve demand from demand or purchase-bid market columns."""
    demand_columns = find_columns(dataframe, ["demand"])
    if demand_columns:
        return demand_columns[0]

    market_purchase_bid_columns = [
        column
        for column in find_columns(dataframe, ["purchase", "bid"])
        if "market" in normalize_column_name(column)
    ]
    if market_purchase_bid_columns:
        return market_purchase_bid_columns[0]
    return first_matching_column(dataframe, ["purchase", "bid"])


def resolve_load_forecast_column(dataframe: pd.DataFrame) -> str:
    """Resolve a load forecast proxy from available IEX market schedule columns."""
    scheduled_market_columns = [
        column
        for column in find_columns(dataframe, ["final", "scheduled", "volume"])
        if "market" in normalize_column_name(column)
    ]
    if scheduled_market_columns:
        return scheduled_market_columns[0]
    return first_matching_column(dataframe, ["mcv"], excluded_tokens=["weather"])


def build_event_timestamp(dataframe: pd.DataFrame) -> pd.Series:
    """Build Feast event timestamps from Date and one-indexed Hour."""
    dates = pd.to_datetime(dataframe["Date"], errors="coerce")
    hours = pd.to_numeric(dataframe["Hour"], errors="coerce")
    timestamps = dates + pd.to_timedelta(hours.sub(1), unit="h")
    invalid_rows = int(timestamps.isna().sum())
    if invalid_rows:
        raise ValueError(f"Found {invalid_rows} row(s) with invalid Date or Hour.")
    return timestamps.dt.tz_localize("UTC")


def numeric_mean(dataframe: pd.DataFrame, columns: list[str], feature_name: str) -> pd.Series:
    """Average a group of numeric source columns into one canonical feature."""
    if not columns:
        raise ValueError(f"No source columns found for feature: {feature_name}")
    return dataframe[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def prepare_feature_store_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical Feast offline source."""
    required_columns = [
        "Date",
        "Hour",
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_48",
        "rolling_mean_24",
        "rolling_std_24",
        "weekday",
        "weekend_flag",
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required feature column(s): {missing}")

    prepared = pd.DataFrame()
    event_timestamp = build_event_timestamp(dataframe)
    prepared["event_timestamp"] = event_timestamp
    prepared["timestamp"] = event_timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    prepared["date"] = event_timestamp.dt.strftime("%Y-%m-%d")
    prepared["hour"] = pd.to_numeric(dataframe["Hour"], errors="coerce").astype("int64")

    prepared["mcp"] = pd.to_numeric(dataframe[resolve_target_column(dataframe)], errors="coerce")
    prepared["demand"] = pd.to_numeric(
        dataframe[resolve_demand_column(dataframe)],
        errors="coerce",
    )
    prepared["load_forecast"] = pd.to_numeric(
        dataframe[resolve_load_forecast_column(dataframe)],
        errors="coerce",
    )
    prepared["renewable_generation"] = numeric_mean(
        dataframe,
        [
            *find_columns(dataframe, ["renewable"]),
            *find_columns(dataframe, ["shortwave", "radiation"]),
            *find_columns(dataframe, ["solar", "radiation"]),
        ],
        "renewable_generation",
    )

    for column in [
        "rolling_mean_24",
        "rolling_std_24",
        "lag_1",
        "lag_2",
        "lag_24",
        "lag_48",
        "weekday",
        "weekend_flag",
    ]:
        prepared[column] = pd.to_numeric(dataframe[column], errors="coerce")

    prepared["temperature"] = numeric_mean(
        dataframe,
        find_columns(dataframe, ["temperature"]),
        "temperature",
    )
    prepared["humidity"] = numeric_mean(
        dataframe,
        find_columns(dataframe, ["humidity"]),
        "humidity",
    )
    prepared["cloud_cover"] = numeric_mean(
        dataframe,
        find_columns(dataframe, ["cloud", "cover"]),
        "cloud_cover",
    )
    prepared["wind_speed"] = numeric_mean(
        dataframe,
        find_columns(dataframe, ["wind", "speed"]),
        "wind_speed",
    )
    prepared["solar_radiation"] = numeric_mean(
        dataframe,
        [
            *find_columns(dataframe, ["shortwave", "radiation"]),
            *find_columns(dataframe, ["solar", "radiation"]),
        ],
        "solar_radiation",
    )

    prepared = prepared[CANONICAL_FEATURE_COLUMNS].copy()
    prepared = prepared.dropna().sort_values("event_timestamp").reset_index(drop=True)
    return prepared


def save_feature_store_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
    parquet_output_path: Path | None,
) -> None:
    """Persist the Feast source CSV and optional Feast parquet mirror."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        dataframe.to_parquet(output_path, index=False)
    else:
        dataframe.to_csv(output_path, index=False)
    logging.info("Saved Feast feature source: %s", output_path)

    if parquet_output_path is not None:
        parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(parquet_output_path, index=False)
        logging.info("Saved Feast parquet source: %s", parquet_output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare IEX power trading features for the local Feast repository."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to engineered_dataset.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the Feast source should be saved.",
    )
    parser.add_argument(
        "--parquet-output-path",
        type=Path,
        default=DEFAULT_PARQUET_OUTPUT_PATH,
        help="Optional parquet mirror used by Feast FileSource materialization.",
    )
    parser.add_argument(
        "--entity-id",
        default=None,
        help="Deprecated compatibility option; timestamp/date/hour keys are used.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare and save Feast feature data."""
    configure_logging()
    args = parse_args()
    engineered = load_engineered_dataset(args.input_path)
    feature_store_df = prepare_feature_store_dataframe(engineered)
    save_feature_store_dataframe(
        feature_store_df,
        args.output_path,
        args.parquet_output_path,
    )
    print(f"Prepared {len(feature_store_df)} Feast feature rows at {args.output_path}.")


if __name__ == "__main__":
    main()
