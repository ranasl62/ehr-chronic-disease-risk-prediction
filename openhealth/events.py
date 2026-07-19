"""In-memory event ring buffer for UI activity feed."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT, REPORTS_DIR

_LOCK = threading.Lock()
_EVENTS: list[dict[str, Any]] = []
_MAX = 200
_JSONL = REPORTS_DIR / "events.jsonl"


def emit(kind: str, message: str, **extra: Any) -> dict[str, Any]:
    ev = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "message": message,
        **extra,
    }
    with _LOCK:
        _EVENTS.append(ev)
        if len(_EVENTS) > _MAX:
            del _EVENTS[: len(_EVENTS) - _MAX]
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with _JSONL.open("a", encoding="utf-8") as f:
            import json

            f.write(json.dumps(ev, default=str) + "\n")
    except Exception:
        pass
    return ev


def list_events(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        return list(_EVENTS[-max(1, min(limit, _MAX)) :])


def clear_events() -> None:
    with _LOCK:
        _EVENTS.clear()
