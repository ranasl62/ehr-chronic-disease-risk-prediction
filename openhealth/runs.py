"""Named experiment runs under reports/runs/."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import MODEL_PATH, PROJECT_ROOT, REPORTS_DIR

RUNS_DIR = REPORTS_DIR / "runs"


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{prefix}"


def run_path(run_id: str) -> Path:
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Invalid run_id")
    return RUNS_DIR / run_id


def ensure_run(run_id: str) -> Path:
    p = run_path(run_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_runs(limit: int = 30) -> list[dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    items = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta = {}
        mp = d / "run_meta.json"
        if mp.is_file():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        items.append(
            {
                "run_id": d.name,
                "path": str(d.relative_to(PROJECT_ROOT)),
                "has_model": (d / "model.pkl").is_file(),
                "meta": meta,
            }
        )
        if len(items) >= limit:
            break
    return items


def write_run_meta(run_id: str, meta: dict[str, Any]) -> Path:
    p = ensure_run(run_id)
    out = p / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def promote_run(run_id: str) -> dict[str, Any]:
    p = run_path(run_id)
    model = p / "model.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"No model.pkl in run {run_id}")
    shutil.copy2(model, MODEL_PATH)
    for name in ("evaluation_report.json", "feature_importance.json", "training_manifest.json"):
        src = p / name
        if src.is_file():
            shutil.copy2(src, REPORTS_DIR / name)
    try:
        from openhealth.config_store import load_config, save_config

        cfg = load_config()
        cfg["active_run_id"] = run_id
        save_config(cfg)
    except Exception:
        pass
    from openhealth.events import emit

    emit("model_promoted", f"Promoted run {run_id} to active model", run_id=run_id)
    return {"run_id": run_id, "model_path": str(MODEL_PATH)}
