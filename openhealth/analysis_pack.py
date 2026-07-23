"""Cohort analysis pack for research methods appendices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import PROJECT_ROOT, REPORTS_DIR
from utils.json_safe import json_safe


def build_analysis_pack(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"dataset not found: {p}")

    df = pd.read_csv(p)
    n_rows = int(len(df))
    label_col = "label" if "label" in df.columns else ("chronic_disease" if "chronic_disease" in df.columns else None)
    n_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else n_rows
    prevalence = None
    if label_col and n_rows:
        y = pd.to_numeric(df[label_col], errors="coerce")
        # patient-level label if longitudinal duplicates
        if "patient_id" in df.columns:
            y = df.groupby("patient_id")[label_col].max()
            y = pd.to_numeric(y, errors="coerce")
        prevalence = float(y.mean()) if y.notna().any() else None

    missingness = {
        str(c): round(float(df[c].isna().mean() * 100), 2) for c in df.columns[:40]
    }
    time_span = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            time_span = f"{ts.min()} → {ts.max()}"

    subgroup_counts: dict[str, Any] = {}
    if "age" in df.columns:
        ages = pd.to_numeric(df["age"], errors="coerce")
        bands = pd.cut(ages, bins=[0, 40, 60, 80, 200], labels=["<=40", "41-60", "61-80", "81+"])
        subgroup_counts["age_band"] = {str(k): int(v) for k, v in bands.value_counts(dropna=False).items()}

    return {
        "kind": "analysis_pack",
        "path": str(p.resolve()),
        "n_rows": n_rows,
        "n_patients": n_patients,
        "n_columns": int(df.shape[1]),
        "label_column": label_col,
        "label_prevalence": prevalence,
        "missingness": missingness,
        "time_span": time_span,
        "subgroup_counts": subgroup_counts,
        "disclaimer": "Descriptive cohort summary for research — not a clinical dashboard.",
    }


def write_analysis_pack(pack: dict[str, Any], *, run_dir: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shared = REPORTS_DIR / "analysis_pack.json"
    payload = json.dumps(json_safe(pack), indent=2)
    shared.write_text(payload, encoding="utf-8")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis_pack.json").write_text(payload, encoding="utf-8")
    return shared
