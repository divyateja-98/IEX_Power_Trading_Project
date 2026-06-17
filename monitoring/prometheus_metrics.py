"""Prometheus instrumentation for the FastAPI serving API."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total number of HTTP requests handled by the FastAPI service.",
    ("method", "path", "status_code"),
)

REQUEST_LATENCY = Histogram(
    "fastapi_request_duration_seconds",
    "HTTP request latency in seconds for the FastAPI service.",
    ("method", "path", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_ERRORS = Counter(
    "fastapi_request_errors_total",
    "Total number of HTTP requests that returned an error status.",
    ("method", "path", "status_code"),
)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Record request count, latency, and errors for Prometheus."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            labels = (
                request.method,
                _route_path(request),
                str(status_code),
            )
            REQUEST_COUNT.labels(*labels).inc()
            REQUEST_LATENCY.labels(*labels).observe(duration)
            if status_code >= 400:
                REQUEST_ERRORS.labels(*labels).inc()


def setup_prometheus_metrics(app: FastAPI) -> None:
    """Attach Prometheus middleware and expose the `/metrics` endpoint."""
    app.add_middleware(PrometheusMetricsMiddleware)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return request.url.path
