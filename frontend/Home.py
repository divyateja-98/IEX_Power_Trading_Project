"""Streamlit frontend entry point for the enterprise project layout."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="IEX Power Trading", layout="wide")

st.title("IEX Power Trading")
st.caption("MCP forecasting, risk analytics, and procurement recommendations")

st.page_link("pages/5_Recommendations.py", label="Procurement Recommendations")
