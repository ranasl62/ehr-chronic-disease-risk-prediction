"""Thin OMOP + FHIR adapters → canonical longitudinal CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from api.data_io import _save_dataframe


def omop_tables_to_longitudinal(
    person: pd.DataFrame,
    measurement: pd.DataFrame | None = None,
    condition: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Minimal OMOP subset:
    person: person_id [, year_of_birth]
    measurement: person_id, measurement_datetime|measurement_date, value_as_number [, measurement_concept_id]
    condition: person_id, condition_start_date, condition_concept_id (used as soft label proxy if needed)
    """
    if "person_id" not in person.columns:
        raise ValueError("OMOP person table requires person_id")
    rows: list[dict[str, Any]] = []
    ages = {}
    if "year_of_birth" in person.columns:
        ages = dict(zip(person["person_id"], person["year_of_birth"]))

    if measurement is not None and len(measurement):
        ts_col = "measurement_datetime" if "measurement_datetime" in measurement.columns else "measurement_date"
        if ts_col not in measurement.columns:
            raise ValueError("measurement needs measurement_datetime or measurement_date")
        for _, r in measurement.iterrows():
            pid = r["person_id"]
            yob = ages.get(pid)
            age = None
            if yob is not None and pd.notna(r[ts_col]):
                try:
                    age = int(str(r[ts_col])[:4]) - int(yob)
                except Exception:
                    age = None
            rows.append(
                {
                    "patient_id": pid,
                    "timestamp": r[ts_col],
                    "lab_value": r.get("value_as_number"),
                    "glucose": r.get("value_as_number"),
                    "age": age,
                    "label": 0,
                }
            )

    if condition is not None and len(condition) and not rows:
        ts_col = "condition_start_date"
        for _, r in condition.iterrows():
            rows.append(
                {
                    "patient_id": r["person_id"],
                    "timestamp": r[ts_col],
                    "icd_code": r.get("condition_concept_id"),
                    "label": 1,
                    "age": None,
                }
            )
    elif condition is not None and len(condition):
        # mark patients with any condition as label 1 on last row
        cond_pids = set(condition["person_id"].tolist())
        for row in rows:
            if row["patient_id"] in cond_pids:
                row["label"] = 1

    if not rows:
        raise ValueError("OMOP import produced no rows — provide measurement and/or condition")
    return pd.DataFrame(rows)


def fhir_bundle_to_longitudinal(bundle: dict[str, Any] | list[Any]) -> pd.DataFrame:
    """Patient + Observation (+ Condition) → longitudinal CSV schema."""
    entries: list[dict] = []
    if isinstance(bundle, list):
        resources = bundle
    elif isinstance(bundle, dict) and bundle.get("resourceType") == "Bundle":
        resources = [e.get("resource") or e for e in bundle.get("entry") or []]
    elif isinstance(bundle, dict) and "resourceType" in bundle:
        resources = [bundle]
    else:
        raise ValueError("Expected FHIR Bundle, resource, or list of resources")

    patients: dict[str, dict] = {}
    observations: list[dict] = []
    conditions: list[dict] = []
    for res in resources:
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")
        if rt == "Patient":
            pid = res.get("id") or "unknown"
            patients[str(pid)] = res
        elif rt == "Observation":
            observations.append(res)
        elif rt == "Condition":
            conditions.append(res)

    rows: list[dict[str, Any]] = []
    cond_subjects = set()
    for c in conditions:
        sub = (c.get("subject") or {}).get("reference", "")
        if "/" in sub:
            cond_subjects.add(sub.split("/")[-1])
        elif sub:
            cond_subjects.add(sub)

    for obs in observations:
        sub = (obs.get("subject") or {}).get("reference", "Patient/unknown")
        pid = sub.split("/")[-1] if "/" in sub else sub
        ts = obs.get("effectiveDateTime") or obs.get("issued") or "1970-01-01"
        val = None
        vq = obs.get("valueQuantity") or {}
        if "value" in vq:
            val = vq["value"]
        rows.append(
            {
                "patient_id": pid,
                "timestamp": ts,
                "lab_value": val,
                "glucose": val,
                "age": None,
                "label": 1 if pid in cond_subjects else 0,
            }
        )

    if not rows and patients:
        for pid in patients:
            rows.append(
                {
                    "patient_id": pid,
                    "timestamp": "1970-01-01",
                    "label": 1 if pid in cond_subjects else 0,
                }
            )
    if not rows:
        raise ValueError("FHIR import produced no Observation/Patient rows")
    return pd.DataFrame(rows)


def import_omop_payload(payload: dict[str, Any], name: str = "omop_import.csv") -> dict[str, Any]:
    person = pd.DataFrame(payload.get("person") or [])
    measurement = pd.DataFrame(payload.get("measurement") or []) if payload.get("measurement") else None
    condition = pd.DataFrame(payload.get("condition_occurrence") or payload.get("condition") or [])
    if condition is not None and len(condition) == 0:
        condition = None
    df = omop_tables_to_longitudinal(person, measurement, condition if condition is not None and len(condition) else None)
    meta = _save_dataframe(df, name)
    meta["source_type"] = "omop"
    return meta


def import_fhir_payload(payload: dict[str, Any] | list[Any], name: str = "fhir_import.csv") -> dict[str, Any]:
    df = fhir_bundle_to_longitudinal(payload)
    meta = _save_dataframe(df, name)
    meta["source_type"] = "fhir"
    return meta


def load_fhir_file(path: Path) -> dict[str, Any] | list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".ndjson":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)
