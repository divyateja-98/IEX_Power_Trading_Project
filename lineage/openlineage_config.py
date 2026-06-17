"""OpenLineage event emission helpers for Marquez."""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
DEFAULT_NAMESPACE = "iex-power-trading"
DEFAULT_MARQUEZ_URL = "http://localhost:5001/api/v1/lineage"
PRODUCER = "https://github.com/OpenLineage/OpenLineage"


@dataclass(frozen=True)
class OpenLineageSettings:
    """Runtime configuration for OpenLineage event emission."""

    url: str
    namespace: str
    enabled: bool
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "OpenLineageSettings":
        return cls(
            url=os.getenv("OPENLINEAGE_URL", DEFAULT_MARQUEZ_URL),
            namespace=os.getenv("OPENLINEAGE_NAMESPACE", DEFAULT_NAMESPACE),
            enabled=os.getenv("OPENLINEAGE_DISABLED", "false").lower()
            not in {"1", "true", "yes"},
            timeout_seconds=float(os.getenv("OPENLINEAGE_TIMEOUT_SECONDS", "2.0")),
        )


@contextmanager
def lineage_run(
    job_name: str,
    *,
    inputs: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Emit START and terminal OpenLineage events around a job."""
    run_id = str(uuid.uuid4())
    settings = OpenLineageSettings.from_env()
    emit_lineage_event(
        "START",
        job_name,
        run_id,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        settings=settings,
    )

    try:
        yield run_id
    except Exception as exc:
        failure_metadata = {**(metadata or {}), "error": repr(exc)}
        emit_lineage_event(
            "FAIL",
            job_name,
            run_id,
            inputs=inputs,
            outputs=outputs,
            metadata=failure_metadata,
            settings=settings,
        )
        raise
    else:
        emit_lineage_event(
            "COMPLETE",
            job_name,
            run_id,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            settings=settings,
        )


def emit_lineage_event(
    event_type: str,
    job_name: str,
    run_id: str,
    *,
    inputs: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    settings: OpenLineageSettings | None = None,
) -> None:
    """Send one OpenLineage event to Marquez."""
    settings = settings or OpenLineageSettings.from_env()
    if not settings.enabled:
        return

    event = build_event(
        event_type=event_type,
        job_name=job_name,
        run_id=run_id,
        namespace=settings.namespace,
        inputs=inputs or [],
        outputs=outputs or [],
        metadata=metadata or {},
    )
    payload = json.dumps(event).encode("utf-8")
    request = Request(
        settings.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            response.read()
    except (OSError, URLError) as exc:
        LOGGER.warning("OpenLineage event emission failed: %s", exc)


def build_event(
    *,
    event_type: str,
    job_name: str,
    run_id: str,
    namespace: str,
    inputs: list[str | Path],
    outputs: list[str | Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an OpenLineage RunEvent payload."""
    now = datetime.now(timezone.utc).isoformat()
    normalized_metadata = _json_safe(metadata)
    return {
        "eventType": event_type,
        "eventTime": now,
        "run": {
            "runId": run_id,
            "facets": {
                "iex_metadata": {
                    "_producer": PRODUCER,
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/BaseFacet.json",
                    "metadata": normalized_metadata,
                },
                "processing_engine": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-0/"
                        "ProcessingEngineRunFacet.json"
                    ),
                    "name": "python",
                    "version": "3",
                    "openlineageAdapterVersion": "custom",
                },
                "nominalTime": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-1/"
                        "NominalTimeRunFacet.json"
                    ),
                    "nominalStartTime": now,
                },
            },
        },
        "job": {
            "namespace": namespace,
            "name": job_name,
            "facets": {
                "sourceCodeLocation": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-1/"
                        "SourceCodeLocationJobFacet.json"
                    ),
                    "type": "git",
                    "url": os.getenv("OPENLINEAGE_SOURCE_URL", "local-workspace"),
                },
                "documentation": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-0/"
                        "DocumentationJobFacet.json"
                    ),
                    "description": json.dumps(normalized_metadata, sort_keys=True),
                },
            },
        },
        "inputs": [_dataset(path, namespace) for path in inputs],
        "outputs": [_dataset(path, namespace) for path in outputs],
        "producer": PRODUCER,
        "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/definitions/RunEvent",
    }


def _dataset(path: str | Path, namespace: str) -> dict[str, Any]:
    path_value = str(path)
    try:
        resolved = Path(path).resolve()
        dataset_namespace = f"file://{socket.gethostname()}"
        dataset_name = resolved.as_posix()
    except OSError:
        dataset_namespace = namespace
        dataset_name = path_value

    return {
        "namespace": dataset_namespace,
        "name": dataset_name,
        "facets": {},
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)
