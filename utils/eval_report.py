"""Load persisted hold-out evaluation JSON (no PHI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import EVALUATION_REPORT_PATH


def load_evaluation_report_safe(path: Path | None = None) -> dict[str, Any] | None:
    p = Path(path or EVALUATION_REPORT_PATH)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def evaluation_aligned_with_manifest(
    artifact_manifest: dict[str, Any] | None,
    evaluation_meta: dict[str, Any] | None,
) -> bool:
    if not artifact_manifest or not evaluation_meta:
        return False
    em = evaluation_meta.get("training_manifest") or {}
    a = artifact_manifest.get("data_sha256")
    b = em.get("data_sha256")
    return bool(a and b and a == b)
