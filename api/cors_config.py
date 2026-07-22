"""CORS allow-list from ``CORS_ORIGINS`` (comma-separated browser origins)."""

from __future__ import annotations

import os

# Local Compose / ``ng serve`` defaults when the env var is unset or empty.
_DEFAULT_ORIGINS = (
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)


def parse_cors_origins(raw: str | None = None) -> list[str]:
    """
    Resolve CORS origins from ``CORS_ORIGINS`` or an explicit string.

    Empty / unset → local workbench defaults. Comma-separated list otherwise,
    e.g. ``https://ehr-risk-framework.larucare.com,http://localhost:8080``.
    """
    if raw is None:
        raw = os.environ.get("CORS_ORIGINS", "")
    origins = [o.strip() for o in str(raw).split(",") if o.strip()]
    return origins or list(_DEFAULT_ORIGINS)
