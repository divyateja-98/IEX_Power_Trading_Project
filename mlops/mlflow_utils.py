"""MLflow utilities for IEX power forecasting experiments."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_NAME = "IEX_Power_Forecasting"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_MODEL_REGISTRY_NAME = "IEX_Power_Forecasting_Model"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Configure MLflow tracking and create/select the experiment."""
    resolved_tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
    )
    mlflow.set_tracking_uri(resolved_tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = None
        if resolved_tracking_uri == DEFAULT_TRACKING_URI:
            artifact_location = DEFAULT_ARTIFACT_DIR.resolve().as_uri()
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(experiment_name)
    logging.info("MLflow tracking URI: %s", resolved_tracking_uri)
    logging.info("MLflow experiment: %s", experiment_name)
    return resolved_tracking_uri


def log_xgboost_artifacts(artifact_paths: Mapping[str, Path]) -> None:
    """Log generated plot artifacts to the active MLflow run."""
    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="plots")
        else:
            logging.warning(
                "Skipping missing MLflow artifact %s: %s",
                artifact_name,
                artifact_path,
            )


def log_xgboost_model(
    *,
    model: XGBRegressor,
    X_test: pd.DataFrame,
    predictions: pd.Series,
    registered_model_name: str = DEFAULT_MODEL_REGISTRY_NAME,
) -> str | None:
    """Log and register the trained XGBoost model in the active MLflow run."""
    signature = infer_signature(X_test, predictions)
    input_example = X_test.head(5)
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )
    logging.info("Registered MLflow model as %s", registered_model_name)
    return get_latest_model_version(registered_model_name)


def get_latest_model_version(model_name: str) -> str | None:
    """Return the highest registered version number for a model."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        return None
    return str(max(versions, key=lambda version: int(version.version)).version)


def promote_latest_model_version(
    model_name: str,
    stage: str = "Production",
    tracking_uri: str | None = None,
) -> str:
    """Promote the newest registered model version to an MLflow stage."""
    resolved_tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI
    )
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise LookupError(f"No registered versions found for {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))
    client.transition_model_version_stage(
        name=model_name,
        version=str(latest_version.version),
        stage=stage,
        archive_existing_versions=True,
    )
    if stage.lower() == "production":
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=str(latest_version.version),
        )
    logging.info(
        "Promoted MLflow model %s version %s to %s",
        model_name,
        latest_version.version,
        stage,
    )
    return str(latest_version.version)
