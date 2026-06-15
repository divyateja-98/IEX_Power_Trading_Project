"""Merge IEX market and weather datasets into a master dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "merge_report.txt"
MARKET_FILE = "IEX_combined.xlsx"
WEATHER_FILE = "IEX_weather.xlsx"
MERGE_KEYS = ["Date", "Hour"]


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load an Excel dataset and standardize column name spacing."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    logging.info("Loading dataset: %s", file_path)
    dataframe = pd.read_excel(file_path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def validate_columns(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """Validate that all required merge columns are present."""
    missing_columns = [column for column in MERGE_KEYS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{dataset_name} is missing required column(s): {missing}")


def remove_duplicate_keys(
    dataframe: pd.DataFrame, dataset_name: str
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate Date-Hour rows and return the number removed."""
    duplicate_count = int(dataframe.duplicated(subset=MERGE_KEYS).sum())
    if duplicate_count:
        logging.info(
            "Removing %s duplicate Date-Hour row(s) from %s",
            duplicate_count,
            dataset_name,
        )

    deduplicated = dataframe.drop_duplicates(subset=MERGE_KEYS, keep="first").copy()
    return deduplicated, duplicate_count


def merge_datasets(
    market_df: pd.DataFrame, weather_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge datasets on Date and Hour and return merge diagnostics."""
    diagnostics = market_df.merge(
        weather_df,
        on=MERGE_KEYS,
        how="outer",
        indicator=True,
        suffixes=("_market", "_weather"),
    )

    merged = market_df.merge(
        weather_df,
        on=MERGE_KEYS,
        how="inner",
        suffixes=("_market", "_weather"),
    )

    return merged, diagnostics


def build_merge_report(
    market_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    market_duplicates_removed: int,
    weather_duplicates_removed: int,
) -> str:
    """Build a text report describing validation, deduplication, and merge results."""
    merge_counts = diagnostics_df["_merge"].value_counts().to_dict()
    matched_rows = int(merge_counts.get("both", 0))
    market_only_rows = int(merge_counts.get("left_only", 0))
    weather_only_rows = int(merge_counts.get("right_only", 0))
    merged_duplicate_keys = int(merged_df.duplicated(subset=MERGE_KEYS).sum())

    report_lines = [
        "IEX Data Merge Report",
        "=====================",
        "",
        "Input files:",
        f"- {MARKET_FILE}",
        f"- {WEATHER_FILE}",
        "",
        "Column validation:",
        f"- Required columns: {', '.join(MERGE_KEYS)}",
        "- Status: Passed",
        "",
        "Duplicate removal:",
        f"- Market duplicates removed: {market_duplicates_removed}",
        f"- Weather duplicates removed: {weather_duplicates_removed}",
        "",
        "Dataset shapes after duplicate removal:",
        f"- Market dataset: {market_df.shape}",
        f"- Weather dataset: {weather_df.shape}",
        f"- Merged master dataset: {merged_df.shape}",
        "",
        "Merge verification:",
        f"- Matched Date-Hour rows: {matched_rows}",
        f"- Market-only Date-Hour rows excluded from inner merge: {market_only_rows}",
        f"- Weather-only Date-Hour rows excluded from inner merge: {weather_only_rows}",
        f"- Duplicate Date-Hour keys in merged dataset: {merged_duplicate_keys}",
        f"- Missing values in merged dataset: {int(merged_df.isna().sum().sum())}",
        "",
        "Merged columns:",
        "\n".join(f"- {column}" for column in merged_df.columns),
    ]
    return "\n".join(report_lines)


def save_outputs(merged_df: pd.DataFrame, report: str, output_path: Path, report_path: Path) -> None:
    """Save merged dataset and merge report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(output_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    logging.info("Saved merged dataset: %s", output_path)
    logging.info("Saved merge report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge IEX market and weather Excel datasets on Date and Hour."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the input Excel files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the merged CSV should be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the merge report should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation, duplicate removal, merge, and output generation."""
    configure_logging()
    args = parse_args()

    market_df = load_dataset(args.data_dir / MARKET_FILE)
    weather_df = load_dataset(args.data_dir / WEATHER_FILE)

    validate_columns(market_df, MARKET_FILE)
    validate_columns(weather_df, WEATHER_FILE)
    logging.info("Column validation passed for required keys: %s", MERGE_KEYS)

    market_df, market_duplicates_removed = remove_duplicate_keys(market_df, MARKET_FILE)
    weather_df, weather_duplicates_removed = remove_duplicate_keys(weather_df, WEATHER_FILE)

    merged_df, diagnostics_df = merge_datasets(market_df, weather_df)
    logging.info("Merge completed with shape: %s", merged_df.shape)

    report = build_merge_report(
        market_df=market_df,
        weather_df=weather_df,
        merged_df=merged_df,
        diagnostics_df=diagnostics_df,
        market_duplicates_removed=market_duplicates_removed,
        weather_duplicates_removed=weather_duplicates_removed,
    )
    print(report)

    save_outputs(merged_df, report, args.output_path, args.report_path)


if __name__ == "__main__":
    main()
