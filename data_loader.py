"""Load IEX datasets and write a data quality summary report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from lineage.openlineage_config import lineage_run

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_summary.txt"
DATA_FILES = ("IEX_combined.xlsx", "IEX_weather.xlsx")


def configure_logging() -> None:
    """Configure console logging for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_excel_file(file_path: Path) -> pd.DataFrame:
    """Load an Excel file into a pandas DataFrame."""
    logging.info("Loading dataset: %s", file_path)
    return pd.read_excel(file_path)


def build_dataset_summary(file_name: str, dataframe: pd.DataFrame) -> str:
    """Build a text summary for one loaded dataset."""
    lines = [
        f"Dataset: {file_name}",
        "-" * (len(file_name) + 9),
        f"Shape: {dataframe.shape}",
        "",
        "Column names:",
        "\n".join(f"- {column}" for column in dataframe.columns),
        "",
        "Data types:",
        dataframe.dtypes.to_string(),
        "",
        "Missing values:",
        dataframe.isna().sum().to_string(),
        "",
    ]
    return "\n".join(lines)


def write_summary_report(summaries: list[str], report_path: Path) -> None:
    """Write all dataset summaries to a report file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(summaries), encoding="utf-8")
    logging.info("Saved summary report: %s", report_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load IEX Excel datasets and generate a data summary report."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing IEX_combined.xlsx and IEX_weather.xlsx.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help="Path where the data summary report should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configured datasets, print summaries, and save the report."""
    configure_logging()
    args = parse_args()

    with lineage_run(
        "data_ingestion",
        inputs=[args.data_dir / file_name for file_name in DATA_FILES],
        outputs=[args.report_path],
        metadata={"stage": "data_ingestion", "source_format": "excel"},
    ):
        summaries: list[str] = []
        for file_name in DATA_FILES:
            file_path = args.data_dir / file_name
            if not file_path.exists():
                logging.warning("File not found, skipping: %s", file_path)
                summaries.append(
                    "\n".join(
                        [
                            f"Dataset: {file_name}",
                            "-" * (len(file_name) + 9),
                            f"Status: File not found at {file_path}",
                        ]
                    )
                )
                continue

            dataframe = load_excel_file(file_path)
            summary = build_dataset_summary(file_name, dataframe)
            print(summary)
            summaries.append(summary)

        write_summary_report(summaries, args.report_path)


if __name__ == "__main__":
    main()
