"""Per-run trust pack: bind metrics, leakage, calibration, SHAP, and hashes to a run_id."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.manifest import sha256_file
from utils.config import REPORTS_DIR
from utils.json_safe import json_safe
from utils.report_images import is_valid_report_png

TRUST_PACK_NAME = "trust_pack.json"

# Artifacts that belong in a research trust pack (per run).
TRUST_ARTIFACTS = (
    "model.pkl",
    "evaluation_report.json",
    "feature_importance.json",
    "training_manifest.json",
    "calibration_holdout.png",
    "leakage_audit.json",
    "shap_summary.png",
    "external_validation_report.json",
    "analysis_pack.json",
)

# Synced to shared reports/ on promote (beyond the classic three JSON files).
PROMOTE_EXTRA = (
    "calibration_holdout.png",
    "leakage_audit.json",
    "shap_summary.png",
    "trust_pack.json",
    "external_validation_report.json",
    "analysis_pack.json",
)


def trust_pack_path(run_dir: Path) -> Path:
    return run_dir / TRUST_PACK_NAME


def _file_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    # PNG figures must be real images (reject 8-byte signature stubs).
    if path.suffix.lower() == ".png" and not is_valid_report_png(path):
        return None
    return {
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _leakage_passed(report: dict[str, Any] | None) -> bool | None:
    if not report:
        return None
    if report.get("split_method") in ("patient_group", "temporal_patient"):
        if not report.get("patient_disjoint_train_test", True):
            return False
    ti = report.get("temporal_integrity") or {}
    if ti and ti.get("passed") is False:
        return False
    return True


def build_trust_pack(run_id: str, run_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in TRUST_ARTIFACTS:
        meta = _file_meta(run_dir / name)
        if meta:
            artifacts[name] = meta

    leakage = None
    lp = run_dir / "leakage_audit.json"
    if lp.is_file():
        try:
            leakage = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            leakage = None

    manifest = None
    mp = run_dir / "training_manifest.json"
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            manifest = None

    has_eval = "evaluation_report.json" in artifacts
    has_leakage = "leakage_audit.json" in artifacts
    has_shap = "shap_summary.png" in artifacts  # only if _file_meta accepted a valid PNG
    has_cal = "calibration_holdout.png" in artifacts
    has_model = "model.pkl" in artifacts
    leakage_ok = _leakage_passed(leakage if isinstance(leakage, dict) else None)

    # Complete enough for a methods pack: model + eval + leakage (SHAP optional but tracked).
    trust_complete = bool(has_model and has_eval and has_leakage and leakage_ok is not False)

    return {
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": (manifest or {}).get("data_sha256") if isinstance(manifest, dict) else None,
        "artifacts": artifacts,
        "flags": {
            "has_model": has_model,
            "has_evaluation": has_eval,
            "has_manifest": "training_manifest.json" in artifacts,
            "has_calibration": has_cal,
            "has_leakage": has_leakage,
            "has_shap": has_shap,
            "has_external_validation": "external_validation_report.json" in artifacts,
            "has_analysis_pack": "analysis_pack.json" in artifacts,
            "leakage_passed": leakage_ok,
            "trust_complete": trust_complete,
        },
    }


def write_trust_pack(run_id: str, run_dir: Path) -> dict[str, Any]:
    pack = build_trust_pack(run_id, run_dir)
    out = trust_pack_path(run_dir)
    out.write_text(json.dumps(json_safe(pack), indent=2), encoding="utf-8")
    return pack


def read_trust_pack(run_dir: Path) -> dict[str, Any] | None:
    p = trust_pack_path(run_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def trust_flags_from_dir(run_dir: Path) -> dict[str, Any]:
    pack = read_trust_pack(run_dir)
    if pack and isinstance(pack.get("flags"), dict):
        return dict(pack["flags"])
    # Fallback without pack file
    return {
        "has_model": (run_dir / "model.pkl").is_file(),
        "has_evaluation": (run_dir / "evaluation_report.json").is_file(),
        "has_manifest": (run_dir / "training_manifest.json").is_file(),
        "has_calibration": is_valid_report_png(run_dir / "calibration_holdout.png"),
        "has_leakage": (run_dir / "leakage_audit.json").is_file(),
        "has_shap": is_valid_report_png(run_dir / "shap_summary.png"),
        "has_external_validation": (run_dir / "external_validation_report.json").is_file(),
        "has_analysis_pack": (run_dir / "analysis_pack.json").is_file(),
        "leakage_passed": None,
        "trust_complete": False,
    }


def mirror_to_shared(src: Path, name: str) -> None:
    """Copy an artifact into shared reports/. Refuse corrupt PNG stubs."""
    if not src.is_file():
        return
    if name.endswith(".png") and not is_valid_report_png(src):
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, REPORTS_DIR / name)


def resolve_active_run_id(explicit: str | None = None) -> str | None:
    if explicit:
        if ".." in explicit or "/" in explicit or "\\" in explicit:
            raise ValueError("Invalid run_id")
        return explicit
    try:
        from openhealth.config_store import load_config

        rid = load_config().get("active_run_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    except Exception:
        pass
    return None
