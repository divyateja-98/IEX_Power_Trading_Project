"""Model training entrypoints and helpers."""

from typing import Any


def train_model(X: Any, y: Any, config: dict[str, Any]) -> Any:
    """Train a forecasting model and return a fitted estimator."""
    raise NotImplementedError
