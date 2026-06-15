"""Runtime environment helpers for local Feast scripts."""

from __future__ import annotations

import os
from pathlib import Path


FEATURE_REPO = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_REPO.parent


def configure_feast_environment() -> Path:
    """Keep Feast/Prometheus metrics files inside the writable project tree."""
    metrics_dir = PROJECT_ROOT / ".feast_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(metrics_dir)
    return metrics_dir
