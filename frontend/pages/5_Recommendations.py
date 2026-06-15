"""Compatibility wrapper for the existing Streamlit recommendation page."""

from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PAGE = PROJECT_ROOT / "dashboard" / "pages" / "5_Recommendations.py"

runpy.run_path(str(LEGACY_PAGE), run_name="__main__")
