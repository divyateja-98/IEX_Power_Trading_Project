"""Clean the merged IEX master dataset and generate a cleaning report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "cleaning_report.txt"


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the master dataset from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {input_path}")

    logging.info("Loading master dataset: %s", input_path)
    return pd.read_csv(input_path)


def remove_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows from the dataset."""
    duplicate_count = int(dataframe.duplicated().sum())
    cleaned = dataframe.drop_duplicates().copy()
    logging.info("Removed %s duplicate row(s)", duplicate_count)
    return cleaned, duplicate_count


def handle_missing_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Apply forward fill followed by median imputation for numeric columns."""
    missing_before = dataframe.isna().sum()

    cleaned = dataframe.ffill()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    medians = cleaned[numeric_columns].median()
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(medians)

    missing_after = cleaned.isna().sum()
    logging.info(
        "Missing values reduced from %s to %s",
        int(missing_before.sum()),
        int(missing_after.sum()),
    )
    return cleaned, missing_before, missing_after


def detect_iqr_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers in numeric columns using the IQR method."""
    numeric_columns = dataframe.select_dtypes(include="number").columns
    outlier_records: list[dict[str, object]] = []

    for column in numeric_columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        outlier_records.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
            }
        )

    outliers = pd.DataFrame(outlier_records)
    logging.info("Detected IQR outliers across %s numeric column(s)", len(outliers))
    return outliers


def build_cleaning_report(
    original_shape: tuple[int, int],
    cleaned_shape: tuple[int, int],
    duplicates_removed: int,
    missing_before: pd.Series,
    missing_after: pd.Series,
    outliers: pd.DataFrame,
) -> str:
    """Build a text report for the cleaning pipeline."""
    outlier_total = 0
    if not outliers.empty:
        outlier_total = int(outliers["outlier_count"].sum())

    report_lines = [
        "IEX Data Cleaning Report",
        "========================",
        "",
        "Dataset shapes:",
        f"- Original dataset: {original_shape}",
        f"- Cleaned dataset: {cleaned_shape}",
        "",
        "Duplicate removal:",
        f"- Duplicate rows removed: {duplicates_removed}",
        "",
        "Missing value handling:",
        "- Method 1: Forward fill",
        "- Method 2: Median imputation for numeric columns",
        f"- Missing values before cleaning: {int(missing_before.sum())}",
        f"- Missing values after cleaning: {int(missing_after.sum())}",
        "",
        "Missing values by column before cleaning:",
        missing_before.to_string(),
        "",
        "Missing values by column after cleaning:",
        missing_after.to_string(),
        "",
        "Outlier detection:",
        "- Method: IQR",
        f"- Total numeric outliers detected: {outlier_total}",
        "",
        "Outliers by numeric column:",
    ]

    if outliers.empty:
        report_lines.append("- No numeric columns found.")
    else:
        report_lines.append(outliers.to_string(index=False))

    return "\n".join(report_lines)


def save_outputs(cleaned: pd.DataFrame, report: str, output_path: Path, report_path: Path) -> None:
    """Save the cleaned dataset and cleaning report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(output_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    logging.info("Saved cleaned dataset: %s", output_path)
    logging.info("Saved cleaning report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean the IEX master dataset and generate a cleaning report."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the merged master dataset CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where cleaned_dataset.csv should be saved.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path where the cleaning report should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete data cleaning pipeline."""
    configure_logging()
    args = parse_args()

    original = load_dataset(args.input_path)
    original_shape = original.shape

    deduplicated, duplicates_removed = remove_duplicates(original)
    cleaned, missing_before, missing_after = handle_missing_values(deduplicated)
    outliers = detect_iqr_outliers(cleaned)

    report = build_cleaning_report(
        original_shape=original_shape,
        cleaned_shape=cleaned.shape,
        duplicates_removed=duplicates_removed,
        missing_before=missing_before,
        missing_after=missing_after,
        outliers=outliers,
    )
    print(report)

    save_outputs(cleaned, report, args.output_path, args.report_path)


if __name__ == "__main__":
    main()
