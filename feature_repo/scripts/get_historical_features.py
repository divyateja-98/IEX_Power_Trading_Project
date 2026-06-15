"""Retrieve point-in-time IEX training features from Feast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


FEATURE_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FEATURE_REPO.parent
DEFAULT_ENTITY_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "feature_store" / "historical_features.parquet"
sys.path.insert(0, str(FEATURE_REPO))

from feast_env import configure_feast_environment  # noqa: E402
from feature_refs import FEATURE_REFS  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve Feast historical features.")
    parser.add_argument("--repo-path", type=Path, default=FEATURE_REPO)
    parser.add_argument("--entity-path", type=Path, default=DEFAULT_ENTITY_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_entity_dataframe(entity_path: Path) -> pd.DataFrame:
    """Load timestamp/date/hour entity rows for point-in-time retrieval."""
    if not entity_path.exists():
        raise FileNotFoundError(f"Entity dataframe not found: {entity_path}")
    if entity_path.suffix.lower() == ".parquet":
        entity_df = pd.read_parquet(entity_path)
    else:
        entity_df = pd.read_csv(entity_path)

    required_columns = ["timestamp", "date", "hour", "event_timestamp"]
    missing_columns = [column for column in required_columns if column not in entity_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required entity column(s): {missing}")

    entity_df = entity_df[required_columns].copy()
    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["event_timestamp"],
        utc=True,
        errors="coerce",
    )
    entity_df = entity_df.dropna(subset=["event_timestamp"]).reset_index(drop=True)
    entity_df["hour"] = pd.to_numeric(entity_df["hour"], errors="coerce").astype("int64")
    return entity_df


def main() -> None:
    """Fetch point-in-time correct training features and save them."""
    configure_feast_environment()
    from feast import FeatureStore

    args = parse_args()
    store = FeatureStore(repo_path=str(args.repo_path))
    entity_df = load_entity_dataframe(args.entity_path)
    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix.lower() == ".csv":
        feature_df.to_csv(args.output_path, index=False)
    else:
        feature_df.to_parquet(args.output_path, index=False)
    print(f"Saved historical Feast features: {args.output_path}")


if __name__ == "__main__":
    main()
