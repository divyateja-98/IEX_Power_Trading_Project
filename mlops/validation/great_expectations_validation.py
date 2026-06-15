"""Compatibility entrypoint for Great Expectations validation.

The production validator lives at the project root in ``validate_data.py`` so
DVC, local runs, and MLOps wrappers all execute the same data-quality rules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validate_data import (
    DEFAULT_CONTEXT_ROOT_DIR,
    DEFAULT_INPUT_PATH,
    DEFAULT_REPORTS_DIR,
    validate_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        "--expectations-dir",
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
    """Run validation and fail the process if any expectation fails."""
    args = parse_args()
    validation_result = validate_dataset(
        input_path=args.input_path,
        context_root_dir=args.expectations_dir,
        reports_dir=args.reports_dir,
    )

    if not validation_result.get("success", False):
        raise SystemExit("Great Expectations validation failed.")

    print("Great Expectations validation passed.")


if __name__ == "__main__":
    main()
