"""In-process background jobs for researcher workbench (train / audit / shap)."""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from utils.config import MODEL_PATH, PROJECT_ROOT, REPORTS_DIR

_LOCK = threading.Lock()
_JOBS: dict[str, "JobRecord"] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="researcher_job")


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def append(self, line: str) -> None:
        self.log.append(line)
        if len(self.log) > 200:
            self.log = self.log[-200:]


def get_job(job_id: str) -> JobRecord | None:
    with _LOCK:
        return _JOBS.get(job_id)


def list_recent_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK:
        items = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)[:limit]
        return [_job_public(j) for j in items]


def _job_public(j: JobRecord) -> dict[str, Any]:
    from utils.json_safe import json_safe

    return json_safe(
        {
            "id": j.id,
            "kind": j.kind,
            "status": j.status,
            "created_at": j.created_at,
            "finished_at": j.finished_at,
            "message": j.message,
            "result": j.result,
            "log_tail": j.log[-40:],
        }
    )


def submit_job(kind: str, fn: Callable[[JobRecord], None]) -> JobRecord:
    job_id = uuid.uuid4().hex[:12]
    rec = JobRecord(id=job_id, kind=kind)
    with _LOCK:
        # Simple rate limit: reject if another job is queued/running
        busy = [j for j in _JOBS.values() if j.status in ("queued", "running")]
        if busy:
            raise RuntimeError(
                f"Another job is {busy[0].status}: {busy[0].id} ({busy[0].kind}) — wait or cancel"
            )
        _JOBS[job_id] = rec

    def _run() -> None:
        if rec.status == "cancelled":
            rec.message = "cancelled before start"
            rec.finished_at = datetime.now(timezone.utc).isoformat()
            return
        rec.status = "running"
        rec.append(f"started {kind}")
        try:
            from openhealth.events import emit

            emit("job_started", f"{kind} started", job_id=job_id, job_kind=kind)
            fn(rec)
            if rec.status == "cancelled":
                rec.message = "cancelled"
                rec.append("cancelled")
            else:
                rec.status = "succeeded"
                rec.message = rec.message or "ok"
                rec.append("succeeded")
                emit("job_succeeded", f"{kind} succeeded", job_id=job_id, job_kind=kind)
        except Exception as exc:
            if rec.status != "cancelled":
                rec.status = "failed"
                rec.message = str(exc)
                rec.append(traceback.format_exc()[-1500:])
                try:
                    from openhealth.events import emit

                    emit("job_failed", str(exc), job_id=job_id, job_kind=kind)
                except Exception:
                    pass
        finally:
            rec.finished_at = datetime.now(timezone.utc).isoformat()

    _EXECUTOR.submit(_run)
    return rec


def cancel_job(job_id: str) -> JobRecord | None:
    with _LOCK:
        rec = _JOBS.get(job_id)
        if not rec:
            return None
        if rec.status == "queued":
            rec.status = "cancelled"
            rec.message = "cancelled"
            rec.finished_at = datetime.now(timezone.utc).isoformat()
            rec.append("cancelled while queued")
        elif rec.status == "running":
            # Best-effort flag; long sklearn fits may finish before check
            rec.status = "cancelled"
            rec.message = "cancel requested"
            rec.append("cancel requested (best-effort)")
        return rec


def run_train_job(rec: JobRecord, params: dict[str, Any]) -> None:
    from openhealth.runs import ensure_run, new_run_id, write_run_meta
    from training.train import run_training
    import shutil

    data_path = Path(params["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"data not found: {data_path}")

    run_id = params.get("run_id") or new_run_id(params.get("model_kind", "train"))
    run_dir = ensure_run(run_id)
    model_out = run_dir / "model.pkl"
    windows = params.get("windows_days")
    windows_days = tuple(windows) if windows else None
    promote = bool(params.get("promote", True))
    rec.append(f"training {params.get('model_kind')} on {data_path.name} → run {run_id}")
    run_training(
        data_path=data_path,
        model_path=model_out,
        model_kind=params.get("model_kind", "logreg"),
        data_format=params.get("data_format", "longitudinal"),
        window_days=int(params.get("window_days", 180)),
        windows_days=windows_days,
        calibrate=bool(params.get("calibrate", False)),
        split_by_patient=bool(params.get("split_by_patient", True)),
        temporal_split=bool(params.get("temporal_split", False)),
        horizon_days=params.get("horizon_days"),
        index_strategy=params.get("index_strategy", "last_event"),
        index_time_col=params.get("index_time_col"),
        feature_inclusive=bool(params.get("feature_inclusive", True)),
        bootstrap_samples=params.get("bootstrap_samples"),
        skip_calibration_plot=False,
        label_col=params.get("label_col"),
        calibration_plot_path=run_dir / "calibration_holdout.png",
    )
    # Promote calibration plot into shared reports/ for workspace checklist + ZIP
    cal_run = run_dir / "calibration_holdout.png"
    if cal_run.is_file():
        shutil.copy2(cal_run, REPORTS_DIR / "calibration_holdout.png")
    # Copy key reports into run dir
    for name in ("evaluation_report.json", "feature_importance.json", "training_manifest.json"):
        src = REPORTS_DIR / name
        if src.is_file():
            shutil.copy2(src, run_dir / name)
    write_run_meta(
        run_id,
        {
            "kind": "train",
            "model_kind": params.get("model_kind"),
            "data_path": str(data_path),
            "promoted": promote,
        },
    )
    from openhealth.trust_pack import write_trust_pack

    write_trust_pack(run_id, run_dir)
    if promote:
        shutil.copy2(model_out, MODEL_PATH)
        from openhealth.runs import promote_run

        # Sync shared reports + active_run_id (includes trust extras)
        try:
            promote_run(run_id)
        except Exception:
            try:
                from openhealth.config_store import load_config, save_config

                cfg = load_config()
                cfg["active_run_id"] = run_id
                save_config(cfg)
            except Exception:
                pass
    try:
        from api.main import get_artifact

        get_artifact.cache_clear()
    except Exception:
        pass
    rec.result = {
        "model_path": str(MODEL_PATH if promote else model_out),
        "run_id": run_id,
        "data_path": str(data_path),
        "promoted": promote,
    }
    rec.message = f"training complete (run {run_id})"


def run_compare_job(rec: JobRecord, params: dict[str, Any]) -> None:
    from openhealth.compare import compare_models
    from openhealth.events import emit

    data_path = Path(params["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"data not found: {data_path}")

    # Default: do not promote best (user promotes explicitly)
    promote_best = bool(params.get("promote_best", False))
    windows = params.get("windows_days")
    rec.append(f"comparing models on {data_path.name} (promote_best={promote_best})")
    summary = compare_models(
        data_path=data_path,
        data_format=params.get("data_format", "longitudinal"),
        models=params.get("models"),
        calibrate=bool(params.get("calibrate", False)),
        split_by_patient=bool(params.get("split_by_patient", True)),
        temporal_split=bool(params.get("temporal_split", False)),
        windows_days=tuple(windows) if windows else (7, 30, 180),
        window_days=int(params.get("window_days", 180)),
        horizon_days=params.get("horizon_days"),
        index_strategy=params.get("index_strategy", "last_event"),
        index_time_col=params.get("index_time_col"),
        feature_inclusive=bool(params.get("feature_inclusive", True)),
        label_col=params.get("label_col"),
        promote_best=promote_best,
    )
    emit("compare_done", f"selected {summary.get('selected_model')}", selected=summary.get("selected_model"))
    try:
        from api.main import get_artifact

        get_artifact.cache_clear()
    except Exception:
        pass
    rec.result = {
        "selected_model": summary.get("selected_model"),
        "comparison": summary.get("comparison"),
        "report": "reports/model_comparison.json",
        "promoted": promote_best,
    }
    rec.message = f"selected {summary.get('selected_model')} (promoted={promote_best})"


def run_fairness_job(rec: JobRecord, params: dict[str, Any]) -> None:
    """Smoke fairness: age bands from artifact hold-out if age-like feature exists."""
    import json

    import joblib
    import numpy as np
    import pandas as pd

    from fairness.bias_metrics import binary_rates_by_group, subgroup_metrics_table
    from training.reproduce_split import split_train_test_from_artifact
    from utils.json_safe import json_safe

    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError("model.pkl missing — train first")
    art = joblib.load(MODEL_PATH)
    _, X_test, _, y_test, _, _ = split_train_test_from_artifact(art)
    model = art["model"]
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    y_true = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)

    groups_path = params.get("groups_path")
    group_col = params.get("group_column", "age_band")
    if groups_path:
        gp = Path(groups_path)
        if not gp.is_absolute():
            gp = PROJECT_ROOT / gp
        gdf = pd.read_csv(gp)
        if len(gdf) != len(y_true):
            out = {
                "skipped": True,
                "reason": "groups CSV length does not match hold-out size; provide aligned subgroups",
            }
        else:
            groups = gdf[group_col].astype(str).to_numpy()
            table = subgroup_metrics_table(y_true, y_pred, proba, groups)
            rates = binary_rates_by_group(y_true, y_pred, groups)
            merged = table.merge(rates[["group", "tpr", "fpr"]], on="group", how="left")
            out = {"skipped": False, "group_column": group_col, "by_group": merged.to_dict(orient="records")}
    else:
        age_cols = [c for c in X_test.columns if "age" in str(c).lower()]
        if not age_cols:
            out = {"skipped": True, "reason": "no age-like feature and no groups_path"}
        else:
            ages = pd.to_numeric(X_test[age_cols[0]], errors="coerce")
            bands = pd.cut(ages, bins=[-np.inf, 50, 65, np.inf], labels=["lt50", "50_65", "ge65"])
            groups = bands.astype(str).fillna("unk").to_numpy()
            table = subgroup_metrics_table(y_true, y_pred, proba, groups)
            rates = binary_rates_by_group(y_true, y_pred, groups)
            merged = table.merge(rates[["group", "tpr", "fpr"]], on="group", how="left")
            out = {
                "skipped": False,
                "group_column": age_cols[0],
                "by_group": merged.to_dict(orient="records"),
            }
    out_path = REPORTS_DIR / "fairness_report.json"
    out_path.write_text(json.dumps(json_safe(out), indent=2), encoding="utf-8")
    rec.result = {"report_path": str(out_path), **out}
    rec.message = "fairness report written" if not out.get("skipped") else f"fairness skipped: {out.get('reason')}"


def run_leakage_audit_job(rec: JobRecord, params: dict[str, Any]) -> None:
    import importlib.util
    import json
    import sys

    from openhealth.runs import ensure_run, run_path
    from openhealth.trust_pack import mirror_to_shared, resolve_active_run_id, write_trust_pack
    from utils.json_safe import json_safe

    audit_path = PROJECT_ROOT / "scripts" / "leakage_audit.py"
    spec = importlib.util.spec_from_file_location("leakage_audit_mod", audit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load leakage_audit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leakage_audit_mod"] = mod
    spec.loader.exec_module(mod)

    run_id = resolve_active_run_id(params.get("run_id"))
    artifact = Path(MODEL_PATH)
    if run_id:
        run_model = run_path(run_id) / "model.pkl"
        if run_model.is_file():
            artifact = run_model

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if params.get("use_artifact") and artifact.is_file():
        report = mod.audit_from_artifact(artifact)
    else:
        data_path = Path(params["data_path"])
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        report = mod.audit_from_raw(
            data_path=data_path,
            data_format=params.get("data_format", "longitudinal"),
            split_by_patient=bool(params.get("split_by_patient", True)),
            temporal_split=bool(params.get("temporal_split", False)),
            test_size=float(params.get("test_size", 0.2)),
            random_state=int(params.get("random_state", 42)),
            window_days=int(params.get("window_days", 180)),
            windows=params.get("windows"),
            horizon_days=params.get("horizon_days"),
            index_strategy=params.get("index_strategy", "last_event"),
            index_time_col=params.get("index_time_col"),
            feature_inclusive=bool(params.get("feature_inclusive", True)),
        )
    payload = json.dumps(json_safe(report), indent=2)
    shared_out = REPORTS_DIR / "leakage_audit.json"
    shared_out.write_text(payload, encoding="utf-8")
    if run_id:
        rd = ensure_run(run_id)
        (rd / "leakage_audit.json").write_text(payload, encoding="utf-8")
        write_trust_pack(run_id, rd)
        mirror_to_shared(rd / "trust_pack.json", "trust_pack.json")
    rec.result = {
        "report_path": str(shared_out),
        "passed": _audit_passed(report),
        "run_id": run_id,
    }
    rec.message = "leakage audit complete"


def _audit_passed(report: dict) -> bool:
    if report.get("split_method") in ("patient_group", "temporal_patient"):
        if not report.get("patient_disjoint_train_test", True):
            return False
    ti = report.get("temporal_integrity") or {}
    if ti and ti.get("passed") is False:
        return False
    return True


def run_hpo_job(rec: JobRecord, params: dict[str, Any]) -> None:
    """Optional light hyperparameter grid (research-scoped)."""
    from training.hpo import run_light_hpo

    data_path = Path(params["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"data not found: {data_path}")

    windows = params.get("windows_days")
    rec.append(f"light HPO for {params.get('model_kind', 'logreg')} on {data_path.name}")
    out = run_light_hpo(
        data_path=data_path,
        model_kind=params.get("model_kind", "logreg"),
        data_format=params.get("data_format", "longitudinal"),
        calibrate=bool(params.get("calibrate", False)),
        split_by_patient=bool(params.get("split_by_patient", True)),
        temporal_split=bool(params.get("temporal_split", False)),
        windows_days=tuple(windows) if windows else (7, 30, 180),
        window_days=int(params.get("window_days", 180)),
        horizon_days=params.get("horizon_days"),
        index_strategy=params.get("index_strategy", "last_event"),
        index_time_col=params.get("index_time_col"),
        feature_inclusive=bool(params.get("feature_inclusive", True)),
        label_col=params.get("label_col"),
        grid=params.get("grid"),
        promote_best=bool(params.get("promote_best", False)),
        max_trials=int(params.get("max_trials", 6)),
    )
    try:
        from api.main import get_artifact

        get_artifact.cache_clear()
    except Exception:
        pass
    rec.result = {
        "report_path": out.get("report_path"),
        "best": out.get("best"),
        "n_trials": out.get("n_trials"),
        "promoted": out.get("promoted"),
    }
    best = out.get("best") or {}
    rec.message = (
        f"HPO done: best roc_auc={best.get('roc_auc')} params={best.get('params')}"
    )


def run_shap_job(rec: JobRecord, params: dict[str, Any]) -> None:
    import joblib

    from explainability.shap_explainer import explain_model
    from openhealth.runs import ensure_run, run_path
    from openhealth.trust_pack import mirror_to_shared, resolve_active_run_id, write_trust_pack
    from training.reproduce_split import split_train_test_from_artifact

    run_id = resolve_active_run_id(params.get("run_id"))
    model_path = Path(MODEL_PATH)
    if run_id:
        run_model = run_path(run_id) / "model.pkl"
        if run_model.is_file():
            model_path = run_model
    if not model_path.is_file():
        raise FileNotFoundError("model.pkl missing — train first")
    art = joblib.load(model_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shared_out = REPORTS_DIR / "shap_summary.png"
    out = Path(params.get("out") or shared_out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if run_id and params.get("out") is None:
        out = ensure_run(run_id) / "shap_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    X_train, X_test, _, _, _, _ = split_train_test_from_artifact(art)
    fe = art.get("feature_engineering") or {}
    explain_model(
        art["model"],
        X_train,
        X_test,
        plot_path=out,
        random_state=int(fe.get("random_state", 42)),
    )
    if run_id:
        # Mirror to shared workspace for checklist / ZIP backward compat
        if out.resolve() != shared_out.resolve():
            mirror_to_shared(out, "shap_summary.png")
        write_trust_pack(run_id, ensure_run(run_id))
        mirror_to_shared(ensure_run(run_id) / "trust_pack.json", "trust_pack.json")
    elif out.resolve() != shared_out.resolve():
        mirror_to_shared(out, "shap_summary.png")
    try:
        from api.main import get_artifact

        get_artifact.cache_clear()
    except Exception:
        pass
    rec.result = {"shap_path": str(out), "run_id": run_id}
    rec.message = "shap summary written"


def run_external_validate_job(rec: JobRecord, params: dict[str, Any]) -> None:
    from openhealth.runs import ensure_run, run_path
    from openhealth.trust_pack import mirror_to_shared, resolve_active_run_id, write_trust_pack
    from training.external_validate import external_validate, write_external_validation_report

    run_id = resolve_active_run_id(params.get("run_id"))
    model_path = Path(MODEL_PATH)
    if run_id:
        rm = run_path(run_id) / "model.pkl"
        if rm.is_file():
            model_path = rm
    data_path = Path(params["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    rec.append(f"external validate on {data_path.name} (run={run_id})")
    report = external_validate(
        artifact_path=model_path,
        data_path=data_path,
        data_format=params.get("data_format", "longitudinal"),
        label_col=params.get("label_col"),
    )
    report["run_id"] = run_id
    run_dir = ensure_run(run_id) if run_id else None
    out = write_external_validation_report(report, run_dir=run_dir)
    if run_id and run_dir is not None:
        write_trust_pack(run_id, run_dir)
        mirror_to_shared(run_dir / "trust_pack.json", "trust_pack.json")
    rec.result = {"report_path": str(out), "metrics": report.get("metrics"), "run_id": run_id}
    rec.message = "external validation complete"
