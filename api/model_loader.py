"""Model loading and inference helpers for the FastAPI serving layer."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

try:
    from mlops.mlflow_utils import DEFAULT_MODEL_REGISTRY_NAME, DEFAULT_TRACKING_URI
except Exception:
    DEFAULT_MODEL_REGISTRY_NAME = "IEX_Power_Forecasting_Model"
    DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "xgboost_model_report.txt"


@dataclass(frozen=True)
class LoadedModel:
    """Container for a loaded model and serving metadata."""

    model: Any
    source: str
    uri: str
    expected_features: list[str] | None
    expected_feature_types: dict[str, str]
    rmse: float | None


class ModelService:
    """Thin prediction service that normalizes inputs for the loaded model."""

    def __init__(self, loaded_model: LoadedModel) -> None:
        self.loaded_model = loaded_model

    @property
    def model_source(self) -> str:
        return self.loaded_model.source

    @property
    def model_uri(self) -> str:
        return self.loaded_model.uri

    @property
    def expected_features(self) -> list[str] | None:
        return self.loaded_model.expected_features

    @property
    def rmse(self) -> float | None:
        return self.loaded_model.rmse

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        """Run one prediction and return prediction plus confidence."""
        frame = self._build_prediction_frame(features)
        raw_prediction = self.loaded_model.model.predict(frame)
        prediction = _extract_single_prediction(raw_prediction)
        confidence = estimate_confidence(prediction=prediction, rmse=self.rmse)
        return prediction, confidence

    def _build_prediction_frame(self, features: dict[str, float]) -> pd.DataFrame:
        if self.expected_features:
            missing = [
                feature for feature in self.expected_features if feature not in features
            ]
            if missing:
                preview = ", ".join(missing[:10])
                suffix = "" if len(missing) <= 10 else f", ... +{len(missing) - 10} more"
                raise ValueError(f"Missing required feature(s): {preview}{suffix}")

            ordered_features = {
                feature: _coerce_feature_value(
                    features[feature],
                    feature,
                    self.loaded_model.expected_feature_types.get(feature),
                )
                for feature in self.expected_features
            }
            return pd.DataFrame([ordered_features], columns=self.expected_features)

        ordered_features = {
            feature: _coerce_float(value, feature) for feature, value in features.items()
        }
        return pd.DataFrame([ordered_features])


def load_model_service(
    *,
    model_registry_name: str = DEFAULT_MODEL_REGISTRY_NAME,
    local_model_path: Path = DEFAULT_LOCAL_MODEL_PATH,
    tracking_uri: str | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> ModelService:
    """Load the latest registered MLflow model, falling back to local XGBoost."""
    rmse = read_report_rmse(report_path)

    try:
        loaded_model = load_latest_mlflow_model(
            model_registry_name=model_registry_name,
            tracking_uri=tracking_uri,
            rmse=rmse,
        )
        LOGGER.info("Loaded model from MLflow registry: %s", loaded_model.uri)
        return ModelService(loaded_model)
    except Exception as exc:
        LOGGER.warning(
            "Unable to load MLflow registered model '%s'; falling back to %s. Error: %s",
            model_registry_name,
            local_model_path,
            exc,
        )

    loaded_model = load_local_model(local_model_path=local_model_path, rmse=rmse)
    LOGGER.info("Loaded local model artifact: %s", loaded_model.uri)
    return ModelService(loaded_model)


def load_latest_mlflow_model(
    *,
    model_registry_name: str,
    tracking_uri: str | None,
    rmse: float | None,
) -> LoadedModel:
    """Load the Production registered model, falling back to highest version."""
    resolved_tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
    )
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    versions = client.search_model_versions(f"name = '{model_registry_name}'")
    if not versions:
        raise LookupError(f"No registered versions found for {model_registry_name}")

    production_versions = [
        version for version in versions if version.current_stage == "Production"
    ]
    selected_version = max(
        production_versions or versions,
        key=lambda version: int(version.version),
    )
    model_uri = f"models:/{model_registry_name}/{selected_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    expected_features = extract_pyfunc_feature_names(model)
    expected_feature_types = extract_pyfunc_feature_types(model)
    return LoadedModel(
        model=model,
        source="mlflow_registry",
        uri=model_uri,
        expected_features=expected_features,
        expected_feature_types=expected_feature_types,
        rmse=rmse,
    )


def load_local_model(*, local_model_path: Path, rmse: float | None) -> LoadedModel:
    """Load the local joblib XGBoost model artifact."""
    if not local_model_path.exists():
        raise FileNotFoundError(f"Local model artifact not found: {local_model_path}")

    model = joblib.load(local_model_path)
    expected_features = extract_sklearn_feature_names(model)
    return LoadedModel(
        model=model,
        source="local_artifact",
        uri=str(local_model_path.resolve()),
        expected_features=expected_features,
        expected_feature_types={},
        rmse=rmse,
    )


def extract_sklearn_feature_names(model: Any) -> list[str] | None:
    """Extract feature names from a scikit-learn compatible model."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return None
    return [str(feature) for feature in feature_names]


def extract_pyfunc_feature_names(model: Any) -> list[str] | None:
    """Extract input names from an MLflow pyfunc model signature."""
    signature = getattr(getattr(model, "metadata", None), "signature", None)
    inputs = getattr(signature, "inputs", None)
    if inputs is None:
        return None

    try:
        names = inputs.input_names()
    except Exception:
        names = [getattr(column, "name", None) for column in inputs]

    cleaned_names = [str(name) for name in names if name]
    return cleaned_names or None


def extract_pyfunc_feature_types(model: Any) -> dict[str, str]:
    """Extract MLflow signature input types keyed by feature name."""
    signature = getattr(getattr(model, "metadata", None), "signature", None)
    inputs = getattr(signature, "inputs", None)
    if inputs is None:
        return {}

    feature_types: dict[str, str] = {}
    for column in inputs:
        name = getattr(column, "name", None)
        data_type = getattr(column, "type", None)
        if name and data_type:
            feature_types[str(name)] = str(data_type).lower()
    return feature_types


def read_report_rmse(report_path: Path) -> float | None:
    """Read RMSE from the training report when present."""
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"RMSE:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    return value if math.isfinite(value) and value >= 0 else None


def estimate_confidence(*, prediction: float, rmse: float | None) -> float:
    """Estimate regression confidence from prediction magnitude and RMSE."""
    if rmse is None or rmse <= 0:
        return 0.5

    scale = max(abs(prediction), rmse, 1.0)
    confidence = 1.0 - min(rmse / scale, 1.0)
    return round(float(max(0.0, min(confidence, 1.0))), 6)


def _coerce_float(value: float, feature_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Feature '{feature_name}' must be a finite number")
    return number


def _coerce_feature_value(value: float, feature_name: str, data_type: str | None) -> float | int:
    number = _coerce_float(value, feature_name)
    if data_type and any(token in data_type for token in ("integer", "int", "long")):
        if not number.is_integer():
            raise ValueError(f"Feature '{feature_name}' must be an integer value")
        return int(number)
    return number


def _extract_single_prediction(raw_prediction: Any) -> float:
    values = np.asarray(raw_prediction).reshape(-1)
    if values.size != 1:
        raise RuntimeError(f"Expected one prediction, received {values.size}")

    prediction = float(values[0])
    if not math.isfinite(prediction):
        raise RuntimeError("Model returned a non-finite prediction")
    return prediction
