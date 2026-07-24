"""Framework extensions: map, config, events, runs, adapters, worklist, fairness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.jobs import cancel_job, run_fairness_job, submit_job, _job_public
from api.security import require_api_key_if_configured
from utils.config import PROJECT_ROOT
from utils.json_safe import json_safe

router = APIRouter(prefix="/v1", tags=["framework"])
AuthDep = Depends(require_api_key_if_configured)


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


class MapPreviewBody(BaseModel):
    path: str


class MapImportBody(BaseModel):
    path: str
    mapping: dict[str, str | None]
    name: str = "mapped_import.csv"
    source_type: str = "byo"


class WorkspaceConfigBody(BaseModel):
    persona: Literal["researcher", "clinical_research"] | None = None
    active_task_id: str | None = None
    windows_days: list[int] | None = None
    horizon_days: int | None = None
    index_strategy: str | None = None
    index_time_col: str | None = None
    model_kind: Literal["logreg", "xgboost", "random_forest", "lightgbm"] | None = None
    compare_models: list[str] | None = None
    calibrate: bool | None = None
    split_by_patient: bool | None = None
    temporal_split: bool | None = None
    feature_inclusive: bool | None = None
    require_api_key: bool | None = None
    disclaimer_ack: bool | None = None
    active_run_id: str | None = None
    data_mode: Literal["synthetic", "real"] | None = None


class TaskUpsertBody(BaseModel):
    id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str
    description: str = ""
    target_column: str | None = "label"
    horizon_days: int | None = None
    index_strategy: str = "last_event"
    index_time_col: str | None = None
    windows_days: list[int] = Field(default_factory=lambda: [7, 30, 180])
    data_format: str = "longitudinal"
    suggested_path: str | None = None
    model_kind: str = "logreg"
    calibrate: bool = False
    split_by_patient: bool = True


class OmopImportBody(BaseModel):
    person: list[dict[str, Any]]
    measurement: list[dict[str, Any]] | None = None
    condition_occurrence: list[dict[str, Any]] | None = None
    name: str = "omop_import.csv"


class FhirImportBody(BaseModel):
    bundle: dict[str, Any] | list[Any]
    name: str = "fhir_import.csv"


class WorklistBody(BaseModel):
    rows: list[dict[str, float]]
    include_explanation: bool = False


class FairnessJobBody(BaseModel):
    groups_path: str | None = None
    group_column: str = "age_band"


@router.post("/datasets/map-preview")
def map_preview(body: MapPreviewBody, _: bool = AuthDep):
    from openhealth.events import emit
    from openhealth.schema_map import map_preview_path

    try:
        out = map_preview_path(_resolve_under_project(body.path))
        emit("map_preview", f"Preview mapping for {body.path}")
        return json_safe(out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/datasets/map-import")
def map_import_route(body: MapImportBody, _: bool = AuthDep):
    from openhealth.events import emit
    from openhealth.schema_map import map_import

    try:
        meta = map_import(
            _resolve_under_project(body.path),
            body.mapping,
            name=body.name,
            source_type=body.source_type,
        )
        emit("dataset_imported", f"Mapped import {body.name}", path=meta.get("path"))
        return json_safe(meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/workspace/config")
def get_workspace_config(_: bool = AuthDep):
    from openhealth.config_store import effective_train_params, load_config

    cfg = load_config()
    return json_safe({"config": cfg, "effective_train": effective_train_params(cfg)})


@router.put("/workspace/config")
def put_workspace_config(body: WorkspaceConfigBody, _: bool = AuthDep):
    from openhealth.config_store import load_config, save_config
    from openhealth.events import emit

    cfg = load_config()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    # allow explicit nulls for horizon via sentinel — only update provided fields
    for k, v in body.model_dump(exclude_unset=True).items():
        cfg[k] = v
    try:
        saved = save_config(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    emit("config_saved", "Workspace config updated", keys=list(updates.keys()))
    return json_safe({"config": saved})


@router.post("/tasks")
def upsert_task(body: TaskUpsertBody, _: bool = AuthDep):
    import yaml

    from openhealth.events import emit
    from openhealth.task_spec import TASKS_DIR

    if ".." in body.id or "/" in body.id:
        raise HTTPException(status_code=400, detail="invalid task id")
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    path = TASKS_DIR / f"{body.id}.yaml"
    doc = {
        "task": {"id": body.id, "name": body.name, "description": body.description},
        "target": {"column": body.target_column},
        "prediction": {
            "horizon_days": body.horizon_days,
            "index_strategy": body.index_strategy,
            "index_time_col": body.index_time_col,
        },
        "features": {"windows_days": body.windows_days},
        "data": {"format": body.data_format, "suggested_path": body.suggested_path},
        "training": {
            "model_kind": body.model_kind,
            "calibrate": body.calibrate,
            "split_by_patient": body.split_by_patient,
        },
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    emit("task_saved", f"Task {body.id} saved", task_id=body.id)
    return {"id": body.id, "path": str(path.relative_to(PROJECT_ROOT))}


@router.get("/events")
def get_events(limit: int = 50, _: bool = AuthDep):
    from openhealth.events import list_events

    return {"events": list_events(limit=limit)}


@router.get("/runs")
def get_runs(limit: int = 30, _: bool = AuthDep):
    from openhealth.runs import list_runs

    return json_safe({"runs": list_runs(limit=limit)})


@router.get("/runs/{run_id}")
def get_run_detail(run_id: str, _: bool = AuthDep):
    from openhealth.runs import get_run

    try:
        return json_safe(get_run(run_id))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/runs/{run_id}/promote")
def promote_run_route(run_id: str, _: bool = AuthDep):
    from openhealth.runs import promote_run

    try:
        return json_safe(promote_run(run_id))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/reports/fairness")
def get_fairness_report(_: bool = AuthDep):
    import json

    from utils.config import REPORTS_DIR

    path = REPORTS_DIR / "fairness_report.json"
    if not path.is_file():
        return {"present": False, "skipped": True, "reason": "no fairness_report.json yet"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"invalid fairness report: {e}") from e
    return json_safe({"present": True, **data})


@router.get("/reports/thresholds")
def get_threshold_points(_: bool = AuthDep):
    """Operating-point table from the active model hold-out (research only)."""
    import joblib
    import numpy as np

    from training.evaluate import threshold_operating_points
    from training.reproduce_split import split_train_test_from_artifact
    from utils.config import MODEL_PATH, REPORTS_DIR

    path = REPORTS_DIR / "threshold_operating_points.json"
    if path.is_file():
        import json

        try:
            return json_safe({"present": True, **json.loads(path.read_text(encoding="utf-8"))})
        except Exception:
            pass
    if not Path(MODEL_PATH).is_file():
        raise HTTPException(status_code=404, detail="model.pkl missing — train first")
    art = joblib.load(MODEL_PATH)
    _, X_test, _, y_test, _, _ = split_train_test_from_artifact(art)
    proba = art["model"].predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
    rows = threshold_operating_points(y_true, proba)
    out = {
        "present": True,
        "threshold": 0.5,
        "points": rows,
        "note": "Research operating points; not clinical decision thresholds.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import json

    from utils.json_safe import json_safe as _js

    path.write_text(json.dumps(_js(out), indent=2), encoding="utf-8")
    return json_safe(out)


@router.post("/jobs/{job_id}/cancel")
def cancel_job_route(job_id: str, _: bool = AuthDep):
    rec = cancel_job(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_public(rec)


@router.post("/jobs/fairness")
def start_fairness(body: FairnessJobBody, _: bool = AuthDep):
    params = body.model_dump()
    try:
        rec = submit_job("fairness", lambda r: run_fairness_job(r, params))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_public(rec)


@router.post("/datasets/from-omop")
def from_omop(body: OmopImportBody, _: bool = AuthDep):
    from openhealth.adapters import import_omop_payload
    from openhealth.events import emit

    try:
        meta = import_omop_payload(body.model_dump(), name=body.name)
        emit("dataset_imported", "OMOP import", path=meta.get("path"), source_type="omop")
        return json_safe(meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/datasets/from-fhir")
def from_fhir(body: FhirImportBody, _: bool = AuthDep):
    from openhealth.adapters import import_fhir_payload
    from openhealth.events import emit

    try:
        meta = import_fhir_payload(body.bundle, name=body.name)
        emit("dataset_imported", "FHIR import", path=meta.get("path"), source_type="fhir")
        return json_safe(meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/worklist/predict")
def worklist_predict(body: WorklistBody, _: bool = AuthDep):
    from openhealth.clinical_audit import append_audit
    from openhealth.config_store import load_config

    cfg = load_config()
    if cfg.get("persona") == "clinical_research" and not cfg.get("disclaimer_ack"):
        raise HTTPException(
            status_code=403,
            detail="clinical_research persona requires disclaimer_ack in workspace config",
        )
    if not body.rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    if len(body.rows) > 200:
        raise HTTPException(status_code=400, detail="max 200 rows per worklist batch")

    from openhealth.api import predict

    results = []
    for i, row in enumerate(body.rows):
        try:
            results.append(predict(row))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"message": str(e), "row_index": i},
            ) from e
    append_audit(
        "worklist_predict",
        {
            "n": len(results),
            "run_id": cfg.get("active_run_id"),
            "persona": cfg.get("persona"),
        },
    )
    return json_safe({"results": results, "n": len(results)})


@router.get("/worklist/audit")
def worklist_audit(limit: int = 50, _: bool = AuthDep):
    from openhealth.clinical_audit import recent_audit

    return {"audit": recent_audit(limit=limit)}


@router.get("/meta/framework")
def framework_meta(_: bool = AuthDep):
    return {
        "supported_models": ["logreg", "random_forest", "xgboost", "lightgbm"],
        "unsupported_models": ["lstm", "transformers"],
        "limitations": "/LIMITATIONS.md",
        "personas": ["researcher", "clinical_research"],
        "adapters": ["csv", "xlsx", "json", "sql", "omop_subset", "fhir_r4_subset"],
        "disclaimer": "For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future.",
    }
