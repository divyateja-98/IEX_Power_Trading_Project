"""Streamlit page for procurement recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation_engine import generate_recommendation


def classify_risk(volatility: float) -> tuple[str, str]:
    """Classify risk level from volatility."""
    if volatility >= 0.35:
        return "High", "error"
    if volatility >= 0.15:
        return "Medium", "warning"
    return "Low", "success"


def display_alert(kind: str, message: str) -> None:
    """Display a Streamlit color-coded alert."""
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.success(message)


st.set_page_config(
    page_title="Procurement Recommendations",
    page_icon="",
    layout="wide",
)

st.title("Procurement Recommendations")

with st.container(border=True):
    with st.form("recommendation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            current_mcp = st.number_input(
                "Current MCP",
                min_value=0.01,
                value=4500.00,
                step=100.00,
                format="%.2f",
            )

        with col2:
            forecast_mcp = st.number_input(
                "Forecast MCP",
                min_value=0.01,
                value=5000.00,
                step=100.00,
                format="%.2f",
            )

        with col3:
            volatility = st.number_input(
                "Volatility",
                min_value=0.00,
                value=0.20,
                step=0.01,
                format="%.4f",
            )

        submitted = st.form_submit_button("Generate Recommendation", type="primary")

if submitted:
    result = generate_recommendation(
        current_mcp=current_mcp,
        forecast_mcp=forecast_mcp,
        volatility=volatility,
    )
    risk_level, alert_kind = classify_risk(volatility)
    recommendation_alert = {
        "BUY NOW": "success",
        "WAIT": "warning",
        "SELL": "error",
    }[result.recommendation]

    with st.container(border=True):
        display_alert(
            recommendation_alert,
            f"Recommendation: {result.recommendation}",
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(
            "Expected Price Change %",
            f"{result.expected_change_pct:.2%}",
            f"{result.expected_change:.2f} Rs/MWh",
        )
        metric_col2.metric("Risk Level", risk_level)
        metric_col3.metric("Volatility", f"{result.volatility:.4f}")

    with st.container(border=True):
        display_alert(alert_kind, f"Risk Level: {risk_level}")
        st.subheader("Business Explanation")
        st.write(result.explanation)

        st.subheader("Input Summary")
        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Current MCP", f"{result.current_mcp:,.2f} Rs/MWh")
        summary_col2.metric("Forecast MCP", f"{result.forecast_mcp:,.2f} Rs/MWh")
