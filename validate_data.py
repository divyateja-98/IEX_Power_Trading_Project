"""Validate IEX power trading data with Great Expectations."""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import great_expectations as gx
import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"
DEFAULT_CONTEXT_ROOT_DIR = PROJECT_ROOT / "expectations"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "validation_reports"
SUITE_NAME = "iex_power_data_quality"
MIN_ROW_COUNT = 1000


def configure_logging() -> None:
    """Configure console logging for validation runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the dataset to validate."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logging.info("Loading dataset for validation: %s", input_path)
    dataframe = pd.read_csv(input_path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def normalize_column_name(column: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return column.lower().replace("_", " ").replace("-", " ")


def find_columns(
    dataframe: pd.DataFrame,
    required_tokens: Iterable[str],
    excluded_tokens: Iterable[str] | None = None,
) -> list[str]:
    """Find columns containing required tokens and no excluded tokens."""
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


def first_existing_column(dataframe: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first candidate column present in the dataframe."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def resolve_mcp_column(dataframe: pd.DataFrame) -> str | None:
    """Resolve the MCP price column from known IEX dataset variants."""
    market_columns = [
        column
        for column in dataframe.columns
        if "mcp" in normalize_column_name(column)
        and "market" in normalize_column_name(column)
    ]
    if market_columns:
        return market_columns[0]

    mcp_columns = find_columns(dataframe, ["mcp"])
    return mcp_columns[0] if mcp_columns else None


def resolve_demand_column(dataframe: pd.DataFrame) -> str | None:
    """Resolve demand from literal demand or purchase-bid market columns."""
    demand_column = first_existing_column(
        dataframe,
        ["Demand", "Demand (MW)", "demand", "demand_mw"],
    )
    if demand_column:
        return demand_column

    market_purchase_bid_columns = [
        column
        for column in find_columns(dataframe, ["purchase", "bid"])
        if "market" in normalize_column_name(column)
    ]
    if market_purchase_bid_columns:
        return market_purchase_bid_columns[0]

    purchase_bid_columns = find_columns(dataframe, ["purchase", "bid"])
    return purchase_bid_columns[0] if purchase_bid_columns else None


def resolve_renewable_columns(dataframe: pd.DataFrame) -> list[str]:
    """Resolve renewable generation or non-negative renewable proxy columns."""
    renewable_generation_columns = [
        column
        for column in dataframe.columns
        if any(
            token in normalize_column_name(column)
            for token in ["renewable", "solar", "wind generation", "generation"]
        )
    ]
    if renewable_generation_columns:
        return renewable_generation_columns

    return [
        column
        for column in dataframe.columns
        if any(
            token in normalize_column_name(column)
            for token in ["shortwave radiation", "wind speed"]
        )
    ]


def create_data_context(context_root_dir: Path) -> None:
    """Create a minimal local Great Expectations context directory."""
    (context_root_dir / "expectations").mkdir(parents=True, exist_ok=True)
    (context_root_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (context_root_dir / "plugins").mkdir(parents=True, exist_ok=True)
    (context_root_dir / "uncommitted").mkdir(parents=True, exist_ok=True)
    (context_root_dir / "uncommitted" / "validations").mkdir(parents=True, exist_ok=True)

    config_variables_path = context_root_dir / "uncommitted" / "config_variables.yml"
    if not config_variables_path.exists():
        config_variables_path.write_text("{}\n", encoding="utf-8")

    config_path = context_root_dir / "great_expectations.yml"
    if not config_path.exists():
        config_path.write_text(
            "\n".join(
                [
                    "config_version: 4.0",
                    "config_variables_file_path: uncommitted/config_variables.yml",
                    "plugins_directory: plugins/",
                    "stores:",
                    "  expectations_store:",
                    "    class_name: ExpectationsStore",
                    "    store_backend:",
                    "      class_name: TupleFilesystemStoreBackend",
                    "      base_directory: expectations/",
                    "  validation_results_store:",
                    "    class_name: ValidationResultsStore",
                    "    store_backend:",
                    "      class_name: TupleFilesystemStoreBackend",
                    "      base_directory: uncommitted/validations/",
                    "  checkpoint_store:",
                    "    class_name: CheckpointStore",
                    "    store_backend:",
                    "      class_name: TupleFilesystemStoreBackend",
                    "      base_directory: checkpoints/",
                    "expectations_store_name: expectations_store",
                    "validation_results_store_name: validation_results_store",
                    "checkpoint_store_name: checkpoint_store",
                    "data_docs_sites:",
                    "  local_site:",
                    "    class_name: SiteBuilder",
                    "    store_backend:",
                    "      class_name: TupleFilesystemStoreBackend",
                    "      base_directory: ../validation_reports/data_docs/",
                    "    site_index_builder:",
                    "      class_name: DefaultSiteIndexBuilder",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def build_validator(dataframe: pd.DataFrame) -> Validator:
    """Create a Great Expectations validator backed by the pandas dataframe."""
    suite = ExpectationSuite(name=SUITE_NAME)
    engine = PandasExecutionEngine()
    batch = Batch(data=dataframe)
    return Validator(
        execution_engine=engine,
        interactive_evaluation=False,
        expectation_suite=suite,
        batches=[batch],
    )


def add_expectations(validator: Validator, dataframe: pd.DataFrame) -> dict[str, object]:
    """Add the IEX power data-quality expectations to the suite."""
    date_column = first_existing_column(dataframe, ["Date"])
    hour_column = first_existing_column(dataframe, ["Hour"])
    mcp_column = resolve_mcp_column(dataframe)
    demand_column = resolve_demand_column(dataframe)
    renewable_columns = resolve_renewable_columns(dataframe)

    missing_critical = []
    if date_column is None:
        missing_critical.append("Date")
    if hour_column is None:
        missing_critical.append("Hour")
    if mcp_column is None:
        missing_critical.append("MCP")
    if demand_column is None:
        missing_critical.append("Demand")
    if not renewable_columns:
        missing_critical.append("Renewable generation or renewable proxy")

    if missing_critical:
        missing = ", ".join(missing_critical)
        raise ValueError(f"Missing critical validation column(s): {missing}")

    validator.expect_table_row_count_to_be_between(min_value=MIN_ROW_COUNT + 1)
    validator.expect_column_to_exist(date_column)
    validator.expect_column_to_exist(hour_column)
    validator.expect_column_to_exist(mcp_column)
    validator.expect_column_to_exist(demand_column)
    validator.expect_column_values_to_not_be_null(mcp_column)
    validator.expect_column_values_to_not_be_null(demand_column)
    validator.expect_column_values_to_be_between(
        mcp_column,
        min_value=0,
        strict_min=True,
    )
    validator.expect_column_values_to_be_between(
        demand_column,
        min_value=0,
        strict_min=True,
    )
    validator.expect_compound_columns_to_be_unique([date_column, hour_column])

    for renewable_column in renewable_columns:
        validator.expect_column_values_to_be_between(
            renewable_column,
            min_value=0,
        )

    return {
        "date_column": date_column,
        "hour_column": hour_column,
        "mcp_column": mcp_column,
        "demand_column": demand_column,
        "renewable_columns": renewable_columns,
    }


def save_expectation_suite(validator: Validator, context_root_dir: Path) -> Path:
    """Save the expectation suite JSON inside the Great Expectations context."""
    suite_dir = context_root_dir / "expectations"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_path = suite_dir / f"{SUITE_NAME}.json"
    suite_path.write_text(
        json.dumps(validator.expectation_suite.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    logging.info("Saved expectation suite: %s", suite_path)
    return suite_path


def build_html_report(
    validation_result: dict[str, object],
    column_mapping: dict[str, object],
) -> str:
    """Build an HTML validation report from Great Expectations results."""
    def cell(value: object) -> str:
        return html.escape("" if value is None else str(value))

    rows = []
    for item in validation_result.get("results", []):
        expectation_config = item.get("expectation_config", {})
        kwargs = expectation_config.get("kwargs", {})
        expectation_type = expectation_config.get("type") or expectation_config.get(
            "expectation_type", "unknown"
        )
        column = kwargs.get("column") or ", ".join(kwargs.get("column_list", []))
        status = "PASSED" if item.get("success") else "FAILED"
        observed = item.get("result", {}).get("observed_value", "")
        unexpected = item.get("result", {}).get("unexpected_count", "")
        rows.append(
            "<tr>"
            f'<td class="{status}">{status}</td>'
            f"<td>{cell(expectation_type)}</td>"
            f"<td>{cell(column)}</td>"
            f"<td>{cell(observed)}</td>"
            f"<td>{cell(unexpected)}</td>"
            "</tr>"
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    success = validation_result.get("success")
    mapping_rows = "".join(
        f"<tr><td>{cell(key)}</td><td>{cell(value)}</td></tr>"
        for key, value in column_mapping.items()
    )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>IEX Power Data Quality Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:32px;color:#1f2937}",
            "table{border-collapse:collapse;width:100%;margin-top:20px}",
            "th,td{border:1px solid #d1d5db;padding:8px;text-align:left}",
            "th{background:#f3f4f6}",
            ".PASSED{color:#047857;font-weight:700}",
            ".FAILED{color:#b91c1c;font-weight:700}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>IEX Power Data Quality Report</h1>",
            f"<p><strong>Suite:</strong> {cell(SUITE_NAME)}</p>",
            f"<p><strong>Overall success:</strong> {cell(success)}</p>",
            f"<p><strong>Generated at:</strong> {cell(generated_at)}</p>",
            "<h2>Resolved Columns</h2>",
            "<table><thead><tr><th>Role</th><th>Column(s)</th></tr></thead>",
            f"<tbody>{mapping_rows}</tbody></table>",
            "<h2>Expectation Results</h2>",
            "<table>",
            "<thead><tr><th>Status</th><th>Expectation</th><th>Column(s)</th>"
            "<th>Observed</th><th>Unexpected Count</th></tr></thead>",
            "<tbody>",
            "\n".join(rows),
            "</tbody>",
            "</table>",
            "</body>",
            "</html>",
        ]
    )


def save_validation_reports(
    validation_result: dict[str, object],
    reports_dir: Path,
    column_mapping: dict[str, object],
) -> tuple[Path, Path]:
    """Save JSON and HTML validation reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "validation_result.json"
    html_path = reports_dir / "validation_report.html"

    report_payload = {
        "suite_name": SUITE_NAME,
        "column_mapping": column_mapping,
        "validation_result": validation_result,
    }
    json_path.write_text(
        json.dumps(report_payload, indent=2, default=str),
        encoding="utf-8",
    )
    html_path.write_text(
        build_html_report(validation_result, column_mapping),
        encoding="utf-8",
    )

    logging.info("Saved validation result: %s", json_path)
    logging.info("Saved HTML validation report: %s", html_path)
    return json_path, html_path


def validate_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    context_root_dir: Path = DEFAULT_CONTEXT_ROOT_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict[str, object]:
    """Validate the dataset, save expectations/reports, and return the result."""
    create_data_context(context_root_dir)
    gx.get_context(context_root_dir=context_root_dir)
    dataframe = load_dataset(input_path)
    validator = build_validator(dataframe)
    column_mapping = add_expectations(validator, dataframe)

    save_expectation_suite(validator, context_root_dir)
    validation_result = validator.validate().to_json_dict()
    save_validation_reports(validation_result, reports_dir, column_mapping)
    return validation_result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate IEX power trading data with Great Expectations."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the dataset CSV to validate.",
    )
    parser.add_argument(
        "--context-root-dir",
        type=Path,
        default=DEFAULT_CONTEXT_ROOT_DIR,
        help="Great Expectations context directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory where validation reports should be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation and fail the process if critical expectations fail."""
    configure_logging()
    args = parse_args()
    validation_result = validate_dataset(
        input_path=args.input_path,
        context_root_dir=args.context_root_dir,
        reports_dir=args.reports_dir,
    )

    if not validation_result.get("success", False):
        raise SystemExit("Great Expectations validation failed.")

    print("Great Expectations validation passed.")


if __name__ == "__main__":
    main()
