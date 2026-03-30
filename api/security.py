"""
Optional API key for production-style deployments.

If environment variable `API_KEY` is unset or empty, all routes remain open (local dev).
If set, protected routes require header `X-API-Key: <value>`.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_api_key_if_configured(x_api_key: str | None = Header(None, alias="X-API-Key")) -> bool:
    expected = os.environ.get("API_KEY", "").strip()
    if not expected:
        return True
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True
