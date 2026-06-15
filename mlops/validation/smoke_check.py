"""Smoke checks for the IEX MLOps project structure and core artifacts."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PATHS = [
    "data/raw/IEX_combined.xlsx",
    "data/raw/IEX_weather.xlsx",
    "data/processed/master_dataset.csv",
    "data/processed/cleaned_dataset.csv",
    "data/processed/engineered_dataset.csv",
    "models/xgboost_model.pkl",
    "dvc.yaml",
    "params.yaml",
    "backend/app/main.py",
    "frontend/Home.py",
]


def main() -> None:
    """Validate that required project artifacts are present."""
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing required artifact(s):\n{missing_list}")

    print("MLOps smoke check passed.")


if __name__ == "__main__":
    main()
