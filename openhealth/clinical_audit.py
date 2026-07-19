"""Clinical-research audit log (local file, prototype only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import REPORTS_DIR

AUDIT_PATH = REPORTS_DIR / "clinical_audit.jsonl"


def append_audit(event: str, payload: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def recent_audit(limit: int = 50) -> list[dict[str, Any]]:
    if not AUDIT_PATH.is_file():
        return []
    lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
