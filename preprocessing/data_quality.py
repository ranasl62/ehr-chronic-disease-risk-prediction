"""Lightweight checks before training on longitudinal or tabular CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def check_longitudinal(df: pd.DataFrame, *, label_candidates: tuple[str, ...] = ("label", "chronic_disease")) -> list[dict[str, Any]]:
    """Return human-readable issues (empty list = OK for minimal training)."""
    issues: list[dict[str, Any]] = []
    if "patient_id" not in df.columns:
        issues.append({"level": "error", "code": "missing_patient_id", "message": "Column patient_id is required."})
    if "timestamp" not in df.columns:
        issues.append({"level": "error", "code": "missing_timestamp", "message": "Column timestamp is required."})
    lc = [c for c in label_candidates if c in df.columns]
    if not lc:
        issues.append(
            {
                "level": "error",
                "code": "missing_label",
                "message": f"Need one of labels: {label_candidates}.",
            }
        )
    if "patient_id" in df.columns and len(df) > 0:
        sizes = df.groupby("patient_id").size()
        if len(sizes) > 1 and bool(sizes.eq(1).all()):
            issues.append(
                {
                    "level": "warning",
                    "code": "sparse_longitudinal",
                    "message": "Each patient has only one row; multi-window aggregates will be thin — consider more visits per patient.",
                }
            )
    if lc:
        ycol = lc[0]
        vc = df.groupby("patient_id")[ycol].nunique()
        if (vc > 1).any():
            issues.append(
                {
                    "level": "warning",
                    "code": "mixed_labels_per_patient",
                    "message": f"Patients with multiple conflicting values in {ycol}; training uses max() per patient.",
                }
            )
    return issues


def check_tabular(df: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    need = ("patient_id", "chronic_disease")
    for c in need:
        if c not in df.columns:
            issues.append({"level": "error", "code": f"missing_{c}", "message": f"Column {c} is required for tabular pipeline."})
    return issues


def assert_no_blocking_errors(issues: list[dict[str, Any]]) -> None:
    errs = [i for i in issues if i.get("level") == "error"]
    if errs:
        raise ValueError("; ".join(e["message"] for e in errs))


def summarize_csv(path: str | Path, *, data_format: str) -> list[dict[str, Any]]:
    p = Path(path)
    df = pd.read_csv(p)
    if data_format == "longitudinal":
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if df["timestamp"].isna().all():
                return [{"level": "error", "code": "bad_timestamp", "message": "No valid timestamps after parse."}]
        return check_longitudinal(df)
    return check_tabular(df)
