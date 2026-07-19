"""Column mapping preview/import onto canonical longitudinal schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from api.data_io import _save_dataframe, ensure_uploads
from openhealth.health import dataset_health_report
from preprocessing.canonical_schema import (
    LONGITUDINAL_ALIASES,
    assert_longitudinal_minimum,
    rename_to_canonical_longitudinal,
)
from utils.config import PROJECT_ROOT

CANONICAL_FIELDS = (
    "patient_id",
    "timestamp",
    "label",
    "chronic_disease",
    "index_time",
    "glucose",
    "blood_pressure",
    "cholesterol",
    "age",
    "lab_value",
    "vital_signs",
    "icd_code",
)


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Map each canonical field → source column name (or None)."""
    inv: dict[str, str] = {}
    for canonical, aliases in LONGITUDINAL_ALIASES.items():
        for a in aliases:
            inv[a.lower().replace(" ", "_")] = canonical
    # also identity
    for c in CANONICAL_FIELDS:
        inv[c] = c

    suggested: dict[str, str | None] = {f: None for f in CANONICAL_FIELDS}
    used: set[str] = set()
    for col in columns:
        key = str(col).strip().lower().replace(" ", "_")
        canon = inv.get(key)
        if canon and suggested.get(canon) is None and col not in used:
            suggested[canon] = col
            used.add(col)
    return suggested


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename source columns to canonical names using user mapping."""
    rename = {src: canon for canon, src in mapping.items() if src}
    # Avoid collisions: drop targets that already exist if renaming onto them
    out = df.rename(columns=rename, copy=True)
    out = rename_to_canonical_longitudinal(out)
    return out


def map_preview_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    cols = [str(c) for c in df.columns]
    suggested = suggest_mapping(cols)
    auto = rename_to_canonical_longitudinal(df)
    errors: list[str] = []
    try:
        assert_longitudinal_minimum(auto, need_label=True)
        auto_ok = True
    except ValueError as e:
        auto_ok = False
        errors.append(str(e))
    return {
        "columns": cols,
        "suggested_mapping": suggested,
        "canonical_fields": list(CANONICAL_FIELDS),
        "auto_alias_ok": auto_ok,
        "errors": errors,
        "n_rows": int(len(df)),
    }


def map_preview_path(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    df = pd.read_csv(p)
    out = map_preview_from_dataframe(df)
    out["path"] = str(p.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in p.resolve().parents else str(p)
    return out


def map_import(
    path: Path | str,
    mapping: dict[str, str | None],
    *,
    name: str = "mapped_import.csv",
    source_type: str = "byo",
) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    df = pd.read_csv(p)
    mapped = apply_mapping(df, mapping)
    try:
        assert_longitudinal_minimum(mapped, need_label=True)
    except ValueError as e:
        raise ValueError(str(e)) from e
    meta = _save_dataframe(mapped, name)
    meta["source_type"] = source_type
    # health on saved file
    dest = PROJECT_ROOT / meta["path"]
    meta["health"] = dataset_health_report(dest).get("health")
    return meta


def enrich_upload_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    return rename_to_canonical_longitudinal(df)
