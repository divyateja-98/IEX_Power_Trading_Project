"""Streamlit frontend for the IEX Power Forecasting FastAPI service."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


# The production FastAPI prediction endpoint for this project.
DEFAULT_API_URL = os.getenv("FASTAPI_PREDICT_URL", "http://127.0.0.1:8002/predict")

# Defaults are based on a representative row from the current feature store.
# The user controls the three business inputs while the remaining features are
# populated automatically to match the MLflow model signature.
DEFAULT_FEATURES: dict[str, float] = {
    "demand": 6771.7,
    "renewable_generation": 0.0,
    "load_forecast": 5054.1,
    "hour": 1.0,
    "temperature": 16.335714285714285,
    "humidity": 75.14285714285714,
    "cloud_cover": 10.071428571428571,
    "wind_speed": 8.285714285714285,
    "solar_radiation": 0.0,
    "weekday": 1.0,
    "weekend_flag": 0.0,
    "lag_1": 2938.66,
    "lag_2": 3735.05,
    "lag_24": 2543.58,
    "lag_48": 2890.56,
    "rolling_mean_24": 5818.813333333333,
    "rolling_std_24": 3171.596210228571,
}


class PredictionApiError(RuntimeError):
    """Raised when the prediction API cannot return a successful response."""


def build_payload(
    *,
    demand: float,
    renewable_generation: float,
    load_forecast: float,
) -> dict[str, Any]:
    """Build the JSON payload expected by the FastAPI `/predict` endpoint."""
    features = DEFAULT_FEATURES.copy()
    features.update(
        {
            "demand": float(demand),
            "renewable_generation": float(renewable_generation),
            "load_forecast": float(load_forecast),
        }
    )

    request_id = datetime.now(timezone.utc).strftime("streamlit-%Y%m%d%H%M%S")
    return {"features": features, "request_id": request_id}


def call_prediction_api(
    *,
    api_url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call FastAPI and return the decoded prediction response."""
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        message = detail or exc.reason or "No response body returned."
        raise PredictionApiError(f"API returned HTTP {exc.code}: {message}") from exc
    except TimeoutError as exc:
        raise PredictionApiError(
            f"Prediction request timed out after {timeout_seconds} seconds."
        ) from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise PredictionApiError(
            f"Could not connect to the prediction API at {api_url}. Details: {reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PredictionApiError("API returned an invalid JSON response.") from exc


def format_number(value: Any, decimals: int = 2) -> str:
    """Format numeric API values for metric display."""
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "Unavailable"


def render_sidebar() -> tuple[str, int]:
    """Render sidebar controls and return API configuration."""
    with st.sidebar:
        st.header("Configuration")
        api_url = st.text_input("FastAPI endpoint", value=DEFAULT_API_URL)
        timeout_seconds = st.slider(
            "Request timeout",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
            help="Maximum number of seconds to wait for the prediction API.",
        )

        st.divider()
        st.subheader("Auto-filled Features")
        st.caption("These values are sent with every request unless changed in code.")
        st.json(
            {
                key: value
                for key, value in DEFAULT_FEATURES.items()
                if key
                not in {"demand", "renewable_generation", "load_forecast"}
            }
        )

    return api_url.strip(), timeout_seconds


def render_prediction_result(result: dict[str, Any]) -> None:
    """Render prediction outputs from the API response."""
    st.success("Prediction completed successfully.")

    prediction_col, confidence_col = st.columns(2)
    prediction_col.metric(
        "Prediction Value",
        format_number(result.get("prediction")),
    )
    confidence_col.metric(
        "Confidence Score",
        f"{float(result.get('confidence', 0.0)):.2%}",
    )

    metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
    metadata_col1.metric("Model Source", str(result.get("model_source", "Unavailable")))
    metadata_col2.metric("Model URI", str(result.get("model_uri", "Unavailable")))
    metadata_col3.metric("Timestamp", str(result.get("timestamp", "Unavailable")))


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="IEX Power Forecasting",
        layout="wide",
    )

    api_url, timeout_seconds = render_sidebar()

    st.title("IEX Power Forecasting")
    st.caption("Forecast MCP through the production FastAPI prediction service.")

    with st.container(border=True):
        st.subheader("Input Form")
        st.warning(
            "Start the FastAPI service on port 8002 before running predictions.",
            
        )

        with st.form("prediction_form"):
            input_col1, input_col2, input_col3 = st.columns(3)

            with input_col1:
                demand = st.number_input(
                    "Demand",
                    min_value=0.0,
                    value=float(DEFAULT_FEATURES["demand"]),
                    step=100.0,
                    format="%.2f",
                )
            with input_col2:
                renewable_generation = st.number_input(
                    "Renewable Generation",
                    min_value=0.0,
                    value=float(DEFAULT_FEATURES["renewable_generation"]),
                    step=100.0,
                    format="%.2f",
                )
            with input_col3:
                load_forecast = st.number_input(
                    "Load Forecast",
                    min_value=0.0,
                    value=float(DEFAULT_FEATURES["load_forecast"]),
                    step=100.0,
                    format="%.2f",
                )

            predict_clicked = st.form_submit_button("Predict", type="primary")

    if not predict_clicked:
        st.info("Enter the input values and click Predict.")
        return

    payload = build_payload(
        demand=demand,
        renewable_generation=renewable_generation,
        load_forecast=load_forecast,
    )

    with st.expander("Request Payload", expanded=False):
        st.json(payload)

    with st.spinner("Calling FastAPI prediction endpoint..."):
        try:
            result = call_prediction_api(
                api_url=api_url,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        except PredictionApiError as exc:
            st.error(str(exc), )
            return

    render_prediction_result(result)

    with st.expander("Response JSON", expanded=False):
        st.json(result)


if __name__ == "__main__":
    main()
