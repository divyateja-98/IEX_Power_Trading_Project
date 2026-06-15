"""Enterprise backend entry point.

This module preserves the existing FastAPI implementation in ``api.main`` while
exposing the backend through the new MLOps-oriented project layout.
"""

from api.main import app

__all__ = ["app"]
