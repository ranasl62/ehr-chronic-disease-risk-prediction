"""
Map heterogeneous EHR / claims exports onto the longitudinal schema expected by
`load_ehr_data` and `clean_longitudinal_ehr`.

Designed for analyst workflows (CMS-style IDs, claim dates, diagnosis codes) —
not a full FHIR mapper. See `docs/data_sources_and_schema.md` for source systems.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Canonical column -> accepted aliases (lowercase keys after normalization)
LONGITUDINAL_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": (
        "patient_id",
        "pat_id",
        "member_id",
        "desynpuf_id",
        "bene_id",
        "subject_id",
        "person_id",
    ),
    "timestamp": (
        "timestamp",
        "charttime",
        "start_date",
        "service_date",
        "claim_dt",
        "claim_date",
        "visit_date",
        "admit_date",
        "dos",
    ),
    "icd_code": (
        "icd_code",
        "icd10",
        "icd9_code",
        "diagnosis_code",
        "dx_code",
        "principal_diagnosis",
    ),
    "lab_value": ("lab_value", "lab_result", "loinc_value", "result_num"),
    "vital_signs": ("vital_signs", "vitals", "hr", "heart_rate"),
    "glucose": ("glucose", "glucose_mg_dl", "bg", "blood_glucose"),
    "blood_pressure": (
        "blood_pressure",
        "bp",
        "systolic_bp",
        "sbp",
        "blood_pressure_systolic",
    ),
    "cholesterol": ("cholesterol", "ldl", "total_cholesterol", "chol"),
    "age": ("age", "age_years", "patient_age"),
    "label": ("label", "outcome", "target", "case_flag"),
    "chronic_disease": ("chronic_disease", "chf_flag", "diabetes_flag"),
}


def _norm_col(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def rename_to_canonical_longitudinal(
    df: pd.DataFrame,
    *,
    alias_map: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Rename columns to canonical names. First matching alias per canonical wins.
    Unmapped columns are left unchanged.
    """
    am = alias_map or LONGITUDINAL_ALIASES
    inv: dict[str, str] = {}
    for canonical, aliases in am.items():
        for a in aliases:
            inv[_norm_col(a)] = canonical

    out = df.copy()
    rename: dict[str, str] = {}
    for c in out.columns:
        nc = _norm_col(str(c))
        if nc in inv:
            rename[c] = inv[nc]
    return out.rename(columns=rename, copy=False)


def assert_longitudinal_minimum(df: pd.DataFrame, *, need_label: bool = True) -> None:
    """Raise ValueError if required columns are missing after normalization."""
    req: Iterable[str] = ("patient_id", "timestamp")
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Run rename_to_canonical_longitudinal first.")
    if need_label and "label" not in df.columns and "chronic_disease" not in df.columns:
        raise ValueError("Need outcome column: 'label' or 'chronic_disease'.")
