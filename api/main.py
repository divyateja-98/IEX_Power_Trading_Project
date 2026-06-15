"""FastAPI application for production model serving."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.model_loader import ModelService, load_model_service
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once during application startup."""
    LOGGER.info("Starting IEX Forecasting API")
    app.state.model_service = load_model_service()
    service: ModelService = app.state.model_service
    LOGGER.info(
        "Model ready. source=%s uri=%s expected_features=%s",
        service.model_source,
        service.model_uri,
        len(service.expected_features or []),
    )
    yield
    LOGGER.info("Stopping IEX Forecasting API")


app = FastAPI(
    title="IEX Power Trading Forecasting API",
    description=(
        "Production model-serving API for IEX MCP forecasting. The service loads "
        "the latest registered MLflow model when available and falls back to the "
        "local XGBoost model artifact under models/."
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "IEX Power Trading Forecasting"},
    openapi_tags=[
        {"name": "health", "description": "Service and model readiness checks."},
        {"name": "prediction", "description": "XGBoost MCP forecasting endpoints."},
        {"name": "metadata", "description": "Serving model metadata."},
    ],
)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    LOGGER.warning("Validation error during request handling: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


def get_model_service() -> ModelService:
    """Return the initialized model service or raise a clear API error."""
    service = getattr(app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return service


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check API and model readiness",
)
def health_check() -> HealthResponse:
    service = getattr(app.state, "model_service", None)
    return HealthResponse(
        status="ok" if service is not None else "degraded",
        model_loaded=service is not None,
        model_source=getattr(service, "model_source", None),
        model_uri=getattr(service, "model_uri", None),
        expected_feature_count=len(service.expected_features)
        if service is not None and service.expected_features is not None
        else None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid feature payload."},
        500: {"model": ErrorResponse, "description": "Inference failure."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
    tags=["prediction"],
    summary="Forecast MCP for one feature row",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    service = get_model_service()
    LOGGER.info(
        "Prediction request received. request_id=%s feature_count=%s",
        request.request_id,
        len(request.features),
    )

    prediction, confidence = service.predict(request.features)
    LOGGER.info(
        "Prediction completed. request_id=%s prediction=%.6f confidence=%.6f",
        request.request_id,
        prediction,
        confidence,
    )
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_source=service.model_source,
        model_uri=service.model_uri,
        request_id=request.request_id,
    )


@app.get(
    "/model",
    response_model=ModelMetadataResponse,
    tags=["metadata"],
    summary="Get loaded model metadata",
)
def model_metadata() -> ModelMetadataResponse:
    service = get_model_service()
    return ModelMetadataResponse(
        model_source=service.model_source,
        model_uri=service.model_uri,
        expected_features=service.expected_features,
        confidence_reference={"rmse": service.rmse},
    )
