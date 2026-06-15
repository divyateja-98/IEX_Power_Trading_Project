"""Retrieve online IEX power trading features from Feast."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


FEATURE_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FEATURE_REPO.parent
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered.csv"
sys.path.insert(0, str(FEATURE_REPO))

from feast_env import configure_feast_environment  # noqa: E402
from feature_refs import FEATURE_REFS  # noqa: E402


def latest_entity_row(source_path: Path) -> dict[str, object]:
    """Build an entity row from the latest available Feast source row."""
    if not source_path.exists():
        raise FileNotFoundError(f"Feast source not found: {source_path}")
    source = pd.read_csv(source_path, usecols=["timestamp", "date", "hour"])
    if source.empty:
        raise ValueError(f"Feast source has no rows: {source_path}")
    row = source.iloc[-1]
    return {
        "timestamp": str(row["timestamp"]),
        "date": str(row["date"]),
        "hour": int(row["hour"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve Feast online features.")
    parser.add_argument("--repo-path", type=Path, default=FEATURE_REPO)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--hour", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Fetch online features for one timestamp/date/hour entity row."""
    configure_feast_environment()
    from feast import FeatureStore

    args = parse_args()
    entity_row = latest_entity_row(args.source_path)
    if args.timestamp is not None:
        entity_row["timestamp"] = args.timestamp
    if args.date is not None:
        entity_row["date"] = args.date
    if args.hour is not None:
        entity_row["hour"] = args.hour

    store = FeatureStore(repo_path=str(args.repo_path))
    features = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[entity_row],
    ).to_dict()
    print(features)


if __name__ == "__main__":
    main()
