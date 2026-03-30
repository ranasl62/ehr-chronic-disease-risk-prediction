"""Optional production controls: payload size and client rate limits."""

from __future__ import annotations

import threading
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject large JSON bodies using Content-Length (FastAPI parses body after)."""

    def __init__(self, app, max_bytes: int = 262_144):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("content-length")
            if cl:
                try:
                    n = int(cl)
                    if n > self.max_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": f"Request body exceeds limit of {self.max_bytes} bytes",
                            },
                        )
                except ValueError:
                    pass
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple fixed-window rate limit per client IP (in-memory; use Redis in multi-worker prod)."""

    def __init__(self, app, per_minute: int = 120):
        super().__init__(app)
        self.per_minute = max(1, int(per_minute))
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            dq = self._windows.setdefault(client, deque())
            while dq and dq[0] < window_start:
                dq.popleft()
            if len(dq) >= self.per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded; retry after a minute."},
                    headers={"Retry-After": "60"},
                )
            dq.append(now)
        return await call_next(request)
