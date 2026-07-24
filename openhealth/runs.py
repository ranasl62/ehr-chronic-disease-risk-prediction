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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _run_summary(d: Path) -> dict[str, Any]:
    from openhealth.trust_pack import trust_flags_from_dir

    meta = _read_json(d / "run_meta.json") or {}
    ev = _read_json(d / "evaluation_report.json") or {}
    metrics = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else None
    flags = trust_flags_from_dir(d)
    return {
        "run_id": d.name,
        "path": str(d.relative_to(PROJECT_ROOT)),
        "has_model": flags.get("has_model", (d / "model.pkl").is_file()),
        "has_evaluation": flags.get("has_evaluation", (d / "evaluation_report.json").is_file()),
        "has_manifest": flags.get("has_manifest", (d / "training_manifest.json").is_file()),
        "has_leakage": bool(flags.get("has_leakage")),
        "has_shap": bool(flags.get("has_shap")),
        "has_calibration": bool(flags.get("has_calibration")),
        "trust_complete": bool(flags.get("trust_complete")),
        "leakage_passed": flags.get("leakage_passed"),
        "trust": flags,
        "meta": meta,
        "metrics": metrics,
        "model_kind": meta.get("model_kind") or (ev.get("meta") or {}).get("model_kind"),
    }


def list_runs(limit: int = 30) -> list[dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    items = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        items.append(_run_summary(d))
        if len(items) >= limit:
            break
    return items


def get_run(run_id: str) -> dict[str, Any]:
    """Full run detail: meta, metrics, manifest excerpt, file list."""
    from openhealth.trust_pack import read_trust_pack

    p = run_path(run_id)
    if not p.is_dir():
        raise FileNotFoundError(f"run not found: {run_id}")
    summary = _run_summary(p)
    files = []
    for child in sorted(p.iterdir()):
        if child.is_file():
            files.append({"name": child.name, "bytes": child.stat().st_size})
    return {
        **summary,
        "evaluation": _read_json(p / "evaluation_report.json"),
        "manifest": _read_json(p / "training_manifest.json"),
        "feature_importance": _read_json(p / "feature_importance.json"),
        "trust_pack": read_trust_pack(p),
        "leakage_audit": _read_json(p / "leakage_audit.json"),
        "files": files,
    }


def write_run_meta(run_id: str, meta: dict[str, Any]) -> Path:
    p = ensure_run(run_id)
    out = p / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out


def promote_run(run_id: str) -> dict[str, Any]:
    from openhealth.trust_pack import PROMOTE_EXTRA, write_trust_pack
    from utils.report_images import is_valid_report_png

    p = run_path(run_id)
    model = p / "model.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"No model.pkl in run {run_id}")
    shutil.copy2(model, MODEL_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation_report.json", "feature_importance.json", "training_manifest.json"):
        src = p / name
        if src.is_file():
            shutil.copy2(src, REPORTS_DIR / name)
    for name in PROMOTE_EXTRA:
        src = p / name
        if not src.is_file():
            continue
        # Never promote magic-only / corrupt PNGs into the shared gallery.
        if name.endswith(".png") and not is_valid_report_png(src):
            from utils.report_images import remove_invalid_report_png

            # Drop any stale corrupt shared copy so status/ZIP stay honest.
            remove_invalid_report_png(REPORTS_DIR / name)
            continue
        shutil.copy2(src, REPORTS_DIR / name)
    write_trust_pack(run_id, p)
    # Refresh shared trust pack copy after rewrite
    tp = p / "trust_pack.json"
    if tp.is_file():
        shutil.copy2(tp, REPORTS_DIR / "trust_pack.json")
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
