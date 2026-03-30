"""Reproducibility manifest written alongside training artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        n = 0
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if max_bytes is not None and n >= max_bytes:
                break
    return h.hexdigest()


def git_revision(fallback: str = "unknown") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return fallback


def build_training_manifest(
    *,
    data_path: Path,
    model_path: Path,
    model_kind: str,
    calibrated: bool,
    split_method: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_path = data_path.resolve()
    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path) if data_path.is_file() else None,
        "data_bytes": data_path.stat().st_size if data_path.is_file() else None,
        "model_path": str(model_path.resolve()),
        "model_kind": model_kind,
        "calibrated": calibrated,
        "split_method": split_method,
        "git_revision": git_revision(),
    }
    if extra:
        manifest.update(extra)
    return manifest
