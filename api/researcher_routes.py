"""Researcher workbench routes: datasets, jobs, reports, workspace status."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from api.data_io import (
    build_results_zip,
    import_file_bytes,
    import_form_rows,
    import_sql,
    profile_dataset,
)
from api.jobs import (
    get_job,
    list_recent_jobs,
    run_compare_job,
    run_leakage_audit_job,
    run_shap_job,
    run_train_job,
    submit_job,
    _job_public,
)
from api.security import require_api_key_if_configured
from utils.config import (
    MODEL_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    TRAINING_MANIFEST_PATH,
)
from utils.eval_report import load_evaluation_report_safe
from utils.json_safe import json_safe

router = APIRouter(prefix="/v1", tags=["researcher"])
AuthDep = Depends(require_api_key_if_configured)

UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9._\-]+$")

_BUNDLED = [
    {
        "id": "ehr_data",
        "label": "Tiny longitudinal demo (10 patients)",
        "path": "data/raw/ehr_data.csv",
        "format": "longitudinal",
        "bundled": True,
        "source_type": "demo",
    },
    {
        "id": "paper_synthetic",
        "label": "Paper synthetic cohort (400 patients)",
        "path": "data/raw/paper_synthetic_cohort.csv",
        "format": "longitudinal",
        "bundled": True,
        "source_type": "synthetic",
        "suggested": {
            "horizon_days": 365,
            "index_strategy": "column",
            "index_time_col": "index_time",
            "windows_days": [7, 30, 180],
        },
    },
    {
        "id": "sample_ehr",
        "label": "Tabular sample (legacy)",
        "path": "data/raw/sample_ehr.csv",
        "format": "tabular",
        "bundled": True,
        "source_type": "demo",
    },
]

_REPORT_ALLOWLIST = {
    "evaluation_report.json",
    "feature_importance.json",
    "training_manifest.json",
    "leakage_audit.json",
    "calibration_holdout.png",
    "shap_summary.png",
    "cv_group_metrics.json",
    "model_comparison.json",
    "fairness_report.json",
}


class TrainJobBody(BaseModel):
    data_path: str = Field(..., description="Relative to repo root")
    data_format: Literal["longitudinal", "tabular"] = "longitudinal"
    model_kind: Literal["logreg", "xgboost", "random_forest", "lightgbm"] = "logreg"
    calibrate: bool = False
    split_by_patient: bool = True
    temporal_split: bool = False
    window_days: int = 180
    windows_days: list[int] | None = Field(default=[7, 30, 180])
    horizon_days: int | None = None
    index_strategy: Literal["last_event", "before_last", "column"] = "last_event"
    index_time_col: str | None = None
    feature_inclusive: bool = True
    bootstrap_samples: int | None = None
    label_col: str | None = None
    task_id: str | None = None
    promote: bool = True
    force: bool = False


class CompareJobBody(BaseModel):
    data_path: str
    data_format: Literal["longitudinal", "tabular"] = "longitudinal"
    calibrate: bool = False
    split_by_patient: bool = True
    temporal_split: bool = False
    window_days: int = 180
    windows_days: list[int] | None = Field(default=[7, 30, 180])
    horizon_days: int | None = None
    index_strategy: Literal["last_event", "before_last", "column"] = "last_event"
    index_time_col: str | None = None
    feature_inclusive: bool = True
    label_col: str | None = None
    promote_best: bool = False
    models: list[str] | None = None
    task_id: str | None = None


class LeakageJobBody(BaseModel):
    use_artifact: bool = True
    data_path: str | None = None
    data_format: Literal["longitudinal", "tabular"] = "longitudinal"
    split_by_patient: bool = True
    temporal_split: bool = False
    windows: str | None = "7,30,180"
    window_days: int = 180
    horizon_days: int | None = None
    index_strategy: Literal["last_event", "before_last", "column"] = "last_event"
    index_time_col: str | None = None
    feature_inclusive: bool = True


class FormImportBody(BaseModel):
    name: str = "form_import.csv"
    rows: list[dict[str, Any]]


class SqlImportBody(BaseModel):
    sql: str
    connection_url: str | None = None
    name: str = "sql_import.csv"


def _resolve_under_project(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    root = PROJECT_ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="path must be under project root") from e
    return p


@router.get("/workspace/status")
def workspace_status(_: bool = AuthDep):
    model_present = Path(MODEL_PATH).is_file() and Path(MODEL_PATH).stat().st_size > 0
    ev = load_evaluation_report_safe()
    leakage = REPORTS_DIR / "leakage_audit.json"
    shap = REPORTS_DIR / "shap_summary.png"
    cal = REPORTS_DIR / "calibration_holdout.png"
    manifest: dict = {}
    if TRAINING_MANIFEST_PATH.is_file():
        try:
            manifest = json.loads(TRAINING_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    datasets_ok = any((PROJECT_ROOT / d["path"]).is_file() for d in _BUNDLED)
    return json_safe(
        {
            "api_ok": True,
            "model_ready": model_present,
            "model_path": str(MODEL_PATH),
            "evaluation_present": bool(ev),
            "metrics": (ev or {}).get("metrics"),
            "leakage_audit_present": leakage.is_file(),
            "shap_present": shap.is_file(),
            "calibration_present": cal.is_file(),
            "demo_datasets_available": datasets_ok,
            "training_manifest": {
                k: manifest.get(k)
                for k in (
                    "generated_at_utc",
                    "data_sha256",
                    "model_kind",
                    "calibrated",
                    "split_method",
                )
                if k in manifest
            },
            "checklist": {
                "api_healthy": True,
                "demo_dataset": datasets_ok,
                "model_trained": model_present,
                "metrics_available": bool(ev),
                "leakage_audited": leakage.is_file(),
                "shap_available": shap.is_file(),
            },
            "import_formats": ["csv", "tsv", "json", "xlsx", "xls", "form", "sql"],
            "download": {"results_zip": "/v1/reports/download.zip"},
            "recent_jobs": list_recent_jobs(5),
        }
    )


@router.get("/datasets")
def list_datasets(_: bool = AuthDep):
    out = []
    for d in _BUNDLED:
        p = PROJECT_ROOT / d["path"]
        item = {**d, "exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0}
        out.append(item)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(UPLOADS_DIR.glob("*.csv")):
        out.append(
            {
                "id": f"upload:{p.name}",
                "label": f"Upload: {p.name}",
                "path": str(p.relative_to(PROJECT_ROOT)),
                "format": "longitudinal",
                "bundled": False,
                "source_type": "byo",
                "exists": True,
                "bytes": p.stat().st_size,
            }
        )
    return {"datasets": out}


@router.get("/tasks")
def list_prediction_tasks(_: bool = AuthDep):
    from openhealth.task_spec import list_tasks

    return {"tasks": [t.to_public() for t in list_tasks()]}


@router.get("/tasks/{task_id}")
def get_prediction_task(task_id: str, _: bool = AuthDep):
    from openhealth.task_spec import load_task

    try:
        return load_task(task_id).to_public()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/datasets/health")
def dataset_health(path: str, _: bool = AuthDep):
    from openhealth.health import dataset_health_report

    try:
        return json_safe(dataset_health_report(_resolve_under_project(path)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), _: bool = AuthDep):
    """Accept csv/tsv/json/xlsx and normalize to CSV under data/uploads/."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")
    name = Path(file.filename).name
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    try:
        return import_file_bytes(name, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/datasets/from-form")
def dataset_from_form(body: FormImportBody, _: bool = AuthDep):
    try:
        return import_form_rows(body.rows, name=body.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/datasets/from-sql")
def dataset_from_sql(body: SqlImportBody, _: bool = AuthDep):
    try:
        return import_sql(body.sql, connection_url=body.connection_url, name=body.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/datasets/profile")
def dataset_profile(
    path: str,
    age_band: str | None = None,
    label: str | None = None,
    patient_id: str | None = None,
    _: bool = AuthDep,
):
    try:
        return json_safe(
            profile_dataset(
                _resolve_under_project(path),
                age_band=age_band,
                label=label,
                patient_id=patient_id,
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/jobs/train")
def start_train(body: TrainJobBody, _: bool = AuthDep):
    from openhealth.config_store import effective_train_params, load_config
    from openhealth.events import emit
    from openhealth.health import dataset_health_report

    params = body.model_dump()
    cfg = load_config()
    # Fill unset fields from workspace config defaults
    eff = effective_train_params(cfg)
    for key in (
        "windows_days",
        "horizon_days",
        "index_strategy",
        "index_time_col",
        "model_kind",
        "calibrate",
        "split_by_patient",
        "temporal_split",
        "feature_inclusive",
    ):
        if params.get(key) is None and key in eff and eff[key] is not None:
            params[key] = eff[key]
    if not params.get("task_id") and eff.get("task_id"):
        params["task_id"] = eff["task_id"]
    if body.task_id:
        from openhealth.task_spec import load_task

        try:
            spec = load_task(body.task_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        tp = spec.to_train_params(body.data_path)
        if body.horizon_days is None:
            params["horizon_days"] = tp.get("horizon_days")
        if body.index_time_col is None:
            params["index_time_col"] = tp.get("index_time_col")
        if body.label_col is None:
            params["label_col"] = tp.get("label_col")
        if body.index_strategy == "last_event" and tp.get("index_strategy"):
            params["index_strategy"] = tp["index_strategy"]
        if tp.get("windows_days"):
            params["windows_days"] = tp["windows_days"]
        params["calibrate"] = body.calibrate or tp.get("calibrate", False)
    path = _resolve_under_project(params["data_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {params['data_path']}")
    # Health gate
    try:
        health = dataset_health_report(path).get("health") or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"health check failed: {e}") from e
    if not health.get("ready_for_training"):
        force = bool(params.get("force"))
        persona = cfg.get("persona", "researcher")
        if persona == "clinical_research" and force:
            raise HTTPException(
                status_code=400,
                detail="clinical_research persona cannot force-train past health blockers",
            )
        if not force:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "dataset not ready for training",
                    "blockers": health.get("blockers"),
                    "warnings": health.get("warnings"),
                },
            )
    params["data_path"] = str(path)
    if params.get("temporal_split"):
        params["split_by_patient"] = False
    try:
        rec = submit_job("train", lambda r: run_train_job(r, params))
        emit("train_queued", f"Train job {rec.id}", job_id=rec.id)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/jobs/compare")
def start_compare(body: CompareJobBody, _: bool = AuthDep):
    from openhealth.config_store import effective_train_params, load_config
    from openhealth.health import dataset_health_report

    params = body.model_dump()
    cfg = load_config()
    eff = effective_train_params(cfg)
    for key in (
        "windows_days",
        "horizon_days",
        "index_strategy",
        "index_time_col",
        "calibrate",
        "split_by_patient",
        "temporal_split",
        "feature_inclusive",
    ):
        if params.get(key) is None and key in eff and eff[key] is not None:
            params[key] = eff[key]
    if not params.get("models") and cfg.get("compare_models"):
        params["models"] = list(cfg["compare_models"])
    if body.task_id:
        from openhealth.task_spec import load_task

        try:
            spec = load_task(body.task_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        tp = spec.to_train_params(body.data_path)
        params.update({k: v for k, v in tp.items() if k != "task_id" and v is not None})
        params["data_path"] = body.data_path or tp["data_path"]
        params["promote_best"] = body.promote_best
    path = _resolve_under_project(params["data_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {params['data_path']}")
    try:
        health = dataset_health_report(path).get("health") or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"health check failed: {e}") from e
    if not health.get("ready_for_training"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "dataset not ready for compare",
                "blockers": health.get("blockers"),
                "warnings": health.get("warnings"),
            },
        )
    params["data_path"] = str(path)
    if params.get("temporal_split"):
        params["split_by_patient"] = False
    try:
        rec = submit_job("compare", lambda r: run_compare_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/jobs/leakage-audit")
def start_leakage(body: LeakageJobBody, _: bool = AuthDep):
    params = body.model_dump()
    if not body.use_artifact:
        if not body.data_path:
            raise HTTPException(status_code=400, detail="data_path required when use_artifact=false")
        params["data_path"] = str(_resolve_under_project(body.data_path))
    try:
        rec = submit_job("leakage_audit", lambda r: run_leakage_audit_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/jobs/shap")
def start_shap(_: bool = AuthDep):
    try:
        rec = submit_job("shap", lambda r: run_shap_job(r, {}))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.get("/jobs/{job_id}")
def job_status(job_id: str, _: bool = AuthDep):
    rec = get_job(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_public(rec)


@router.get("/jobs")
def jobs_list(_: bool = AuthDep):
    return {"jobs": list_recent_jobs(20)}


@router.get("/reports/summary")
def reports_summary(_: bool = AuthDep):
    leakage = None
    lp = REPORTS_DIR / "leakage_audit.json"
    if lp.is_file():
        try:
            leakage = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            leakage = None
    ev = load_evaluation_report_safe()
    feature_importance = None
    fip = REPORTS_DIR / "feature_importance.json"
    if fip.is_file():
        try:
            feature_importance = json.loads(fip.read_text(encoding="utf-8"))
        except Exception:
            feature_importance = None
    comparison = None
    cmp = REPORTS_DIR / "model_comparison.json"
    if cmp.is_file():
        try:
            comparison = json.loads(cmp.read_text(encoding="utf-8"))
        except Exception:
            comparison = None
    files = []
    for name in sorted(_REPORT_ALLOWLIST):
        p = REPORTS_DIR / name
        if p.is_file():
            files.append(
                {"name": name, "bytes": p.stat().st_size, "url": f"/v1/reports/file/{name}"}
            )
    return json_safe(
        {
            "metrics": (ev or {}).get("metrics"),
            "evaluation_generated_at_utc": (ev or {}).get("generated_at_utc"),
            "leakage_audit": leakage,
            "feature_importance": feature_importance,
            "model_comparison": comparison,
            "files": files,
            "download_zip": "/v1/reports/download.zip",
        }
    )


@router.get("/reports/download.zip")
def reports_download_zip(_: bool = AuthDep):
    data = build_results_zip()
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="ehr_risk_results_pack.zip"'},
    )


@router.get("/reports/file/{name}")
def reports_file(name: str, _: bool = AuthDep):
    if name not in _REPORT_ALLOWLIST or not _SAFE_NAME.match(name):
        raise HTTPException(status_code=404, detail="file not allowed")
    path = REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file missing")
    media = "application/json" if name.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media, filename=name)
