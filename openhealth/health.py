"""Dataset health report — readiness + leakage heuristics for research ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from api.data_io import profile_dataset
from utils.config import PROJECT_ROOT


def dataset_health_report(path: Path | str, *, task_id: str | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    profile = profile_dataset(p)
    df = pd.read_csv(p)

    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    def _check(name: str, ok: bool, detail: str, *, blocking: bool = False) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "blocking": blocking})
        if not ok and blocking:
            blockers.append(f"{name}: {detail}")
        elif not ok:
            warnings.append(f"{name}: {detail}")

    has_pid = "patient_id" in df.columns
    _check("patient_id", has_pid, "required column present" if has_pid else "missing patient_id", blocking=True)

    label_col = profile.get("label_column")
    _check(
        "label",
        label_col is not None,
        f"using {label_col}" if label_col else "no label / chronic_disease column",
        blocking=True,
    )

    has_ts = "timestamp" in df.columns
    _check(
        "timestamp",
        has_ts,
        "present (longitudinal-ready)" if has_ts else "absent — tabular-only or add timestamp",
        blocking=False,
    )

    if task_id:
        try:
            from openhealth.task_spec import load_task

            spec = load_task(task_id)
            req = spec.required_columns()
            missing = [c for c in req if c not in df.columns]
            # label may be chronic_disease alias
            if missing and "label" in missing and "chronic_disease" in df.columns:
                missing = [c for c in missing if c != "label"]
            ok_task = len(missing) == 0
            detail = (
                f"task `{task_id}` columns ok ({', '.join(req)})"
                if ok_task
                else f"task `{task_id}` missing required columns: {missing} "
                f"(need index/horizon/label contract: {req})"
            )
            _check("task_required_columns", ok_task, detail, blocking=True)
            if spec.index_strategy == "column" and spec.index_time_col:
                _check(
                    "task_index_time",
                    spec.index_time_col in df.columns,
                    f"index_time_col `{spec.index_time_col}` "
                    + ("present" if spec.index_time_col in df.columns else "missing for horizon-safe labeling"),
                    blocking=spec.index_time_col not in df.columns,
                )
        except Exception as e:
            _check("task_required_columns", False, f"could not load task `{task_id}`: {e}", blocking=True)

    if has_pid:
        dup_rows = int(df.duplicated().sum())
        _check("duplicate_rows", dup_rows == 0, f"{dup_rows} exact duplicate rows", blocking=False)
        if has_ts:
            key_dups = int(df.duplicated(subset=["patient_id", "timestamp"]).sum())
            _check(
                "duplicate_events",
                key_dups == 0,
                f"{key_dups} duplicate patient_id+timestamp events",
                blocking=False,
            )

    missing_overall = float(df.isna().mean().mean() * 100) if len(df) else 100.0
    _check(
        "missingness",
        missing_overall < 40.0,
        f"overall missing {missing_overall:.1f}%",
        blocking=missing_overall >= 80.0,
    )

    temporal_ok = True
    temporal_detail = "n/a"
    if has_pid and has_ts:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = int(ts.isna().sum())
        temporal_ok = bad == 0
        temporal_detail = "all timestamps parseable" if temporal_ok else f"{bad} unparseable timestamps"
        if temporal_ok and len(df):
            temporal_detail = "PASS"
    _check("temporal_integrity", temporal_ok, temporal_detail, blocking=not temporal_ok and has_ts)

    # Leakage heuristics (not a full audit)
    leakage_risk = "LOW"
    leakage_notes: list[str] = []
    if label_col and has_ts and "index_time" not in df.columns:
        leakage_notes.append("No index_time column — prefer horizon + index for incident labels")
        leakage_risk = "MEDIUM"
    futureish = [c for c in df.columns if str(c).lower() in ("outcome_time", "label_time", "future_label")]
    if futureish:
        leakage_notes.append(f"Suspicious future columns: {futureish}")
        leakage_risk = "HIGH"
    if label_col and has_ts and "index_time" in df.columns:
        leakage_notes.append("index_time present — good for horizon-safe labeling")
        if leakage_risk == "MEDIUM":  # pragma: no cover -- unreachable: MEDIUM requires missing index_time
            leakage_risk = "LOW"

    n_patients = profile.get("n_patients") or 0
    ready = len(blockers) == 0 and (n_patients is None or n_patients >= 2)
    if isinstance(n_patients, int) and 0 < n_patients < 50:
        warnings.append(
            f"tiny_cohort: {n_patients} patients — metrics may be unstable (recommend ≥50)"
        )

    return {
        **profile,
        "health": {
            "patients": n_patients,
            "features": profile.get("n_columns"),
            "missing_pct_overall": round(missing_overall, 2),
            "temporal_integrity": "PASS" if temporal_ok else "FAIL",
            "leakage_risk": leakage_risk,
            "leakage_notes": leakage_notes,
            "ready_for_training": bool(ready),
            "tiny_cohort": bool(isinstance(n_patients, int) and 0 < n_patients < 50),
            "task_id": task_id,
            "blockers": blockers,
            "warnings": warnings,
            "checks": checks,
        },
    }
