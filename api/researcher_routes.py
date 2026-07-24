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
    build_methods_markdown,
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
    run_external_validate_job,
    run_hpo_job,
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
    resolve_training_data_path,
)
from utils.eval_report import load_evaluation_report_safe
from utils.json_safe import json_safe
from utils.report_images import is_valid_report_png

router = APIRouter(prefix="/v1", tags=["researcher"])
AuthDep = Depends(require_api_key_if_configured)

UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9._\-]+$")

_BUNDLED = [
    {
        "id": "ehr_data",
        "label": "Tiny longitudinal demo (10 patients)",
        "path": "data/demo/ehr_data.csv",
        "format": "longitudinal",
        "bundled": True,
        "source_type": "demo",
        "category": "demo",
    },
    {
        "id": "paper_synthetic",
        "label": "Paper synthetic cohort (N≈3000 events)",
        "path": "data/raw/paper_synthetic_cohort.csv",
        "format": "longitudinal",
        "bundled": True,
        "source_type": "synthetic",
        "category": "demo",
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
        "path": "data/demo/sample_ehr.csv",
        "format": "tabular",
        "bundled": True,
        "source_type": "demo",
        "category": "demo",
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
    "hpo_report.json",
    "threshold_operating_points.json",
    "trust_pack.json",
    "external_validation_report.json",
    "analysis_pack.json",
}


def _csv_header_columns(data_path: Path) -> set[str] | None:
    try:
        import pandas as pd

        header = pd.read_csv(data_path, nrows=0)
        return set(header.columns.astype(str))
    except Exception:
        return None


def _reconcile_merged_column_strategy(
    params: dict[str, Any],
    data_path: Path,
    *,
    explicit_column: bool,
) -> None:
    """
    Task presets often set index_strategy=column + index_time.

    When that merge lands on a CSV without the column (tiny demo / tabular) and the
    caller did not explicitly request column strategy, fall back to last_event
    instead of failing. Explicit column requests still hard-fail in validation.
    """
    if explicit_column:
        return
    strategy = params.get("index_strategy") or "last_event"
    if strategy != "column":
        return
    need = params.get("index_time_col") or "index_time"
    cols = _csv_header_columns(data_path)
    if cols is None:
        return
    if need not in cols:
        params["index_strategy"] = "last_event"
        params["index_time_col"] = None


def _validate_index_column_or_clear(params: dict[str, Any], data_path: Path) -> None:
    """
    Fail fast when strategy is explicitly ``column`` but the CSV lacks the column.
    Otherwise clear a stale ``index_time_col`` so background jobs do not crash.
    """
    strategy = params.get("index_strategy") or "last_event"
    col = params.get("index_time_col")
    if strategy != "column" and not col:
        return
    cols = _csv_header_columns(data_path)
    if cols is None:
        return
    need = col or "index_time"
    if strategy == "column":
        if need not in cols:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"index_strategy='column' requires column {need!r} in the dataset. "
                    "Use last_event/before_last for demos without index_time, "
                    "or switch to paper_synthetic / a cohort that includes index_time."
                ),
            )
        params["index_time_col"] = need
        return
    if col and col not in cols:
        # Stale suggestion from another dataset — train path will ignore it too.
        params["index_time_col"] = None


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
    run_id: str | None = None


class ShapJobBody(BaseModel):
    run_id: str | None = None


class ExternalValidateJobBody(BaseModel):
    data_path: str
    data_format: Literal["longitudinal", "tabular"] = "longitudinal"
    run_id: str | None = None
    label_col: str | None = None


class HpoJobBody(BaseModel):
    data_path: str
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
    label_col: str | None = None
    task_id: str | None = None
    promote_best: bool = False
    max_trials: int = Field(default=6, ge=1, le=12)
    grid: list[dict[str, Any]] | None = None


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
    shap_ok = is_valid_report_png(shap)
    cal_ok = is_valid_report_png(cal)
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
            "shap_present": shap_ok,
            "calibration_present": cal_ok,
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
                "shap_available": shap_ok,
            },
            "import_formats": ["csv", "tsv", "json", "xlsx", "xls", "form", "sql"],
            "download": {"results_zip": "/v1/reports/download.zip"},
            "recent_jobs": list_recent_jobs(5),
        }
    )


@router.get("/datasets")
def list_datasets(
    include_demo: bool = True,
    _: bool = AuthDep,
):
    """List datasets. Bundled demos live under data/demo/ (+ paper synthetic in data/raw/).

    Set ``include_demo=false`` to show only user imports under ``data/uploads/``.
    """
    out = []
    if include_demo:
        for d in _BUNDLED:
            p = PROJECT_ROOT / d["path"]
            if p.is_file() or p.is_symlink():
                out.append(
                    {
                        **d,
                        "exists": p.is_file(),
                        "bytes": p.stat().st_size if p.is_file() else 0,
                    }
                )
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(UPLOADS_DIR.glob("*.csv")):
        try:
            exists = p.is_file()
            nbytes = p.stat().st_size if exists else 0
        except OSError:
            exists = False
            nbytes = 0
        out.append(
            {
                "id": f"upload:{p.name}",
                "label": f"Upload: {p.name}",
                "path": str(p.relative_to(PROJECT_ROOT)),
                "format": "longitudinal",
                "bundled": False,
                "source_type": "byo",
                "category": "user",
                "exists": exists,
                "bytes": nbytes,
            }
        )
    return {
        "datasets": out,
        "include_demo": include_demo,
        "demo_root": "data/demo",
        "uploads_root": "data/uploads",
    }


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
def dataset_health(path: str, task_id: str | None = None, _: bool = AuthDep):
    from openhealth.health import dataset_health_report

    try:
        return json_safe(dataset_health_report(_resolve_under_project(path), task_id=task_id))
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


@router.delete("/datasets")
def delete_dataset(path: str, _: bool = AuthDep):
    """Delete a demo or uploaded dataset file (CSV under data/demo, data/uploads, or data/raw)."""
    from api.data_io import delete_dataset_file

    try:
        return delete_dataset_file(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
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
    path = _resolve_under_project(str(resolve_training_data_path(params["data_path"])))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {params['data_path']}")
    # Health gate (task-aware so index_time / required columns match the job)
    try:
        health = dataset_health_report(path, task_id=params.get("task_id")).get("health") or {}
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
                    "hint": (
                        "Use paper_synthetic (or another cohort with index_time) for horizon "
                        "tasks, or switch to the custom / last_event task for the tiny demo."
                    ),
                },
            )
    params["data_path"] = str(path)
    if params.get("temporal_split"):
        params["split_by_patient"] = False
    _reconcile_merged_column_strategy(
        params, path, explicit_column=body.index_strategy == "column"
    )
    _validate_index_column_or_clear(params, path)
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
    path = _resolve_under_project(str(resolve_training_data_path(params["data_path"])))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {params['data_path']}")
    try:
        health = dataset_health_report(path, task_id=params.get("task_id")).get("health") or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"health check failed: {e}") from e
    if not health.get("ready_for_training"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "dataset not ready for compare",
                "blockers": health.get("blockers"),
                "warnings": health.get("warnings"),
                "hint": (
                    "Use paper_synthetic for index_time tasks, or custom/last_event for the tiny demo."
                ),
            },
        )
    params["data_path"] = str(path)
    if params.get("temporal_split"):
        params["split_by_patient"] = False
    _reconcile_merged_column_strategy(
        params, path, explicit_column=body.index_strategy == "column"
    )
    _validate_index_column_or_clear(params, path)
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
        params["data_path"] = str(
            _resolve_under_project(str(resolve_training_data_path(body.data_path)))
        )
    try:
        rec = submit_job("leakage_audit", lambda r: run_leakage_audit_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/jobs/shap")
def start_shap(body: ShapJobBody | None = None, _: bool = AuthDep):
    params = (body or ShapJobBody()).model_dump()
    try:
        rec = submit_job("shap", lambda r: run_shap_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/jobs/external-validate")
def start_external_validate(body: ExternalValidateJobBody, _: bool = AuthDep):
    path = _resolve_under_project(str(resolve_training_data_path(body.data_path)))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {body.data_path}")
    params = body.model_dump()
    params["data_path"] = str(path)
    try:
        rec = submit_job("external_validate", lambda r: run_external_validate_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.get("/reports/analysis-pack")
def reports_analysis_pack(path: str, run_id: str | None = None, _: bool = AuthDep):
    from openhealth.analysis_pack import build_analysis_pack, write_analysis_pack
    from openhealth.runs import ensure_run
    from openhealth.trust_pack import mirror_to_shared, resolve_active_run_id, write_trust_pack

    try:
        resolved = _resolve_under_project(path)
        pack = build_analysis_pack(resolved)
        rid = resolve_active_run_id(run_id)
        run_dir = ensure_run(rid) if rid else None
        write_analysis_pack(pack, run_dir=run_dir)
        if rid and run_dir is not None:
            write_trust_pack(rid, run_dir)
            mirror_to_shared(run_dir / "trust_pack.json", "trust_pack.json")
        return json_safe(pack)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/jobs/hpo")
def start_hpo(body: HpoJobBody, _: bool = AuthDep):
    from openhealth.health import dataset_health_report

    params = body.model_dump()
    if body.task_id:
        from openhealth.task_spec import load_task

        try:
            spec = load_task(body.task_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        tp = spec.to_train_params(body.data_path)
        for k, v in tp.items():
            if k != "task_id" and v is not None and params.get(k) in (None, "", []):
                params[k] = v
        if not params.get("model_kind") and tp.get("model_kind"):
            params["model_kind"] = tp["model_kind"]
    path = _resolve_under_project(str(resolve_training_data_path(params["data_path"])))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"data not found: {params['data_path']}")
    try:
        health = dataset_health_report(path, task_id=params.get("task_id")).get("health") or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"health check failed: {e}") from e
    if not health.get("ready_for_training"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "dataset not ready for HPO",
                "blockers": health.get("blockers"),
                "warnings": health.get("warnings"),
                "hint": (
                    "Use paper_synthetic for index_time tasks, or custom/last_event for the tiny demo."
                ),
            },
        )
    params["data_path"] = str(path)
    if params.get("temporal_split"):
        params["split_by_patient"] = False
    _reconcile_merged_column_strategy(
        params, path, explicit_column=body.index_strategy == "column"
    )
    _validate_index_column_or_clear(params, path)
    try:
        rec = submit_job("hpo", lambda r: run_hpo_job(r, params))
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
    fairness = None
    fp = REPORTS_DIR / "fairness_report.json"
    if fp.is_file():
        try:
            fairness = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            fairness = None
    hpo = None
    hp = REPORTS_DIR / "hpo_report.json"
    if hp.is_file():
        try:
            hpo = json.loads(hp.read_text(encoding="utf-8"))
        except Exception:
            hpo = None
    thresholds = None
    tp = REPORTS_DIR / "threshold_operating_points.json"
    if tp.is_file():
        try:
            thresholds = json.loads(tp.read_text(encoding="utf-8"))
        except Exception:
            thresholds = None
    curves = (ev or {}).get("curves")
    bootstrap_cis = (ev or {}).get("bootstrap_cis")
    quality_note = (ev or {}).get("quality_note")
    files = []
    for name in sorted(_REPORT_ALLOWLIST):
        p = REPORTS_DIR / name
        if not p.is_file():
            continue
        # Do not advertise corrupt / magic-only PNG stubs as figures.
        if name.endswith(".png") and not is_valid_report_png(p):
            continue
        files.append(
            {"name": name, "bytes": p.stat().st_size, "url": f"/v1/reports/file/{name}"}
        )
    return json_safe(
        {
            "metrics": (ev or {}).get("metrics"),
            "evaluation_generated_at_utc": (ev or {}).get("generated_at_utc"),
            "threshold": (ev or {}).get("threshold"),
            "curves": curves,
            "bootstrap_cis": bootstrap_cis,
            "quality_note": quality_note,
            "leakage_audit": leakage,
            "feature_importance": feature_importance,
            "model_comparison": comparison,
            "fairness": fairness,
            "hpo": hpo,
            "thresholds": thresholds,
            "files": files,
            "download_zip": "/v1/reports/download.zip",
        }
    )


@router.get("/reports/curves")
def reports_curves(run_id: str | None = None, _: bool = AuthDep):
    """ROC / PR / calibration curve points for Chart.js (research plots)."""
    root = REPORTS_DIR
    if run_id:
        from openhealth.runs import run_path

        try:
            root = run_path(run_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not root.is_dir():
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    ep = root / "evaluation_report.json"
    if not ep.is_file():
        raise HTTPException(
            status_code=404,
            detail="evaluation_report.json missing — train a model first",
        )
    try:
        ev = json.loads(ep.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid evaluation report: {e}") from e
    curves = ev.get("curves")
    if not curves:
        raise HTTPException(
            status_code=404,
            detail="curves not in evaluation report — retrain to regenerate paper plots",
        )
    return json_safe(
        {
            "run_id": run_id,
            "curves": curves,
            "bootstrap_cis": ev.get("bootstrap_cis"),
            "quality_note": ev.get("quality_note"),
            "metrics": ev.get("metrics"),
        }
    )


@router.get("/reports/download.zip")
def reports_download_zip(run_id: str | None = None, _: bool = AuthDep):
    try:
        data = build_results_zip(run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    fname = f"ehr_risk_results_{run_id}.zip" if run_id else "ehr_risk_results_pack.zip"
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/reports/methods.md")
def reports_methods_md(run_id: str | None = None, _: bool = AuthDep):
    """Short Markdown methods note from active run reports (research honesty tone)."""
    try:
        text = build_methods_markdown(run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(
        content=text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="methods.md"'},
    )


@router.get("/reports/file/{name}")
def reports_file(name: str, _: bool = AuthDep):
    if name not in _REPORT_ALLOWLIST or not _SAFE_NAME.match(name):
        raise HTTPException(status_code=404, detail="file not allowed")
    path = REPORTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file missing")
    if name.endswith(".png") and not is_valid_report_png(path):
        raise HTTPException(
            status_code=404,
            detail="figure corrupt or incomplete — regenerate SHAP / calibration",
        )
    media = "application/json" if name.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media, filename=name)
