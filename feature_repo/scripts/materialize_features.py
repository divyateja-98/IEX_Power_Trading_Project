"""Apply the Feast repository and materialize IEX features to the online store."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FEATURE_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FEATURE_REPO.parent
sys.path.insert(0, str(FEATURE_REPO))

from feast_env import configure_feast_environment  # noqa: E402


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an optional ISO timestamp as timezone-aware UTC."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Materialize IEX Feast features.")
    parser.add_argument("--repo-path", type=Path, default=FEATURE_REPO)
    parser.add_argument(
        "--start-ts",
        default=None,
        help="Optional ISO start timestamp. Defaults to the earliest source timestamp.",
    )
    parser.add_argument(
        "--end-ts",
        default=None,
        help="Optional ISO end timestamp. Defaults to now.",
    )
    return parser.parse_args()


def infer_start_timestamp(source_path: Path) -> datetime:
    """Infer the earliest event timestamp from the Feast CSV source."""
    import pandas as pd

    if not source_path.exists():
        raise FileNotFoundError(f"Feast source not found: {source_path}")
    source = pd.read_csv(source_path, usecols=["event_timestamp"])
    timestamps = pd.to_datetime(source["event_timestamp"], utc=True, errors="coerce")
    if timestamps.isna().all():
        raise ValueError(f"No valid event_timestamp values found in {source_path}")
    return timestamps.min().to_pydatetime()


def main() -> None:
    """Apply Feast definitions and materialize offline features."""
    configure_feast_environment()
    from feast import FeatureStore

    args = parse_args()
    feast_command = shutil.which("feast")
    if feast_command is None:
        raise RuntimeError("Feast CLI not found. Install Feast with `pip install feast`.")

    subprocess.run([feast_command, "-c", str(args.repo_path), "apply"], check=True)
    store = FeatureStore(repo_path=str(args.repo_path))

    source_path = PROJECT_ROOT / "data" / "processed" / "feature_engineered.csv"
    start_ts = parse_timestamp(args.start_ts) or infer_start_timestamp(source_path)
    end_ts = parse_timestamp(args.end_ts) or datetime.now(timezone.utc)
    if end_ts <= start_ts:
        raise ValueError("end-ts must be later than start-ts.")

    store.materialize(start_date=start_ts, end_date=end_ts)
    print(f"Materialized IEX features from {start_ts.isoformat()} to {end_ts.isoformat()}.")


if __name__ == "__main__":
    main()
