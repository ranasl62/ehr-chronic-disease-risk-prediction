"""Request logging with request IDs (no raw PHI in structured fields)."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("ehr_api")

_AUDIT_PATH = os.environ.get("AUDIT_LOG_JSONL", "").strip()


def _should_audit(path: str, method: str) -> bool:
    if method != "POST":
        return False
    return any(
        p in path
        for p in (
            "/v1/predict",
            "/predict",
            "/explain",
        )
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_id=%s path=%s method=%s", rid, request.url.path, request.method)
            raise
        dur_ms = (time.perf_counter() - start) * 1000
        status = getattr(response, "status_code", "?")
        log.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            rid,
            request.method,
            request.url.path,
            status,
            dur_ms,
        )
        if _AUDIT_PATH and _should_audit(request.url.path, request.method):
            rec = {
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(dur_ms, 3),
                "client_ip": request.client.host if request.client else None,
            }
            try:
                with open(_AUDIT_PATH, "a", encoding="utf-8") as af:
                    af.write(json.dumps(rec, default=str) + "\n")
            except OSError as e:
                log.warning("audit log write failed: %s", e)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Clinical-Disclaimer"] = "research-prototype-not-for-diagnostic-use"
        return response


def configure_api_logging(level: int = logging.INFO) -> None:
    if log.handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(level)
