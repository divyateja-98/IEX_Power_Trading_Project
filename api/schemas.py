"""Pydantic schemas for the IEX forecasting serving API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Single-row model inference request."""

    features: dict[str, float] = Field(
        ...,
        description=(
            "Feature values keyed by the exact training feature names. Extra "
            "features are ignored when the loaded model exposes its expected "
            "feature list."
        ),
        example={
            "hour": 12,
            "demand": 15000.0,
            "renewable_generation": 4200.0,
            "load_forecast": 14850.0,
            "rolling_mean_24": 5050.0,
            "rolling_std_24": 120.0,
            "lag_1": 5200.0,
            "lag_2": 5150.0,
            "lag_24": 5000.0,
            "lag_48": 4980.0,
            "temperature": 31.5,
            "humidity": 62.0,
            "cloud_cover": 24.0,
            "wind_speed": 12.4,
            "solar_radiation": 640.0,
            "weekday": 0,
            "weekend_flag": 0,
        },
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller-provided identifier returned in the response.",
        max_length=128,
    )

    @validator("features")
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features must contain at least one feature value")
        return value


class PredictionResponse(BaseModel):
    """Single-row model inference response."""

    prediction: float = Field(..., description="Forecasted MCP value.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic confidence score from 0 to 1. It is derived from the "
            "training RMSE when available and is not a calibrated probability."
        ),
    )
    model_source: str = Field(
        ...,
        description="Source used to load the model: mlflow_registry or local_artifact.",
    )
    model_uri: str = Field(..., description="Resolved MLflow URI or local model path.")
    request_id: str | None = Field(default=None)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC response timestamp.",
    )


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field(..., description="ok when the API process is reachable.")
    model_loaded: bool
    model_source: str | None = None
    model_uri: str | None = None
    expected_feature_count: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str


class ModelMetadataResponse(BaseModel):
    """Loaded model metadata for serving diagnostics."""

    model_source: str
    model_uri: str
    expected_features: list[str] | None = None
    confidence_reference: dict[str, Any] = Field(default_factory=dict)
