"""
Assemble longitudinal patient timelines from MIMIC-style exports (CSV or DataFrames).

Expects aligned time columns: prefer `charttime` for clinical events. Use after running
queries in `sql/feature_queries.sql` and exporting to Parquet/CSV.
"""

from __future__ import annotations

import pandas as pd


def build_patient_timeline(
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    admissions: pd.DataFrame,
    *,
    patient_key: str = "subject_id",
    lab_time_col: str = "charttime",
    vital_time_col: str = "charttime",
) -> pd.DataFrame:
    """
    Outer-merge labs and vitals on patient + time, then left-join admission context.

    All frames must contain `patient_key`. Time columns are coerced to `event_time`.
    """
    labs = labs.copy()
    vitals = vitals.copy()
    if lab_time_col not in labs.columns:
        raise ValueError(f"labs missing {lab_time_col}")
    if vital_time_col not in vitals.columns:
        raise ValueError(f"vitals missing {vital_time_col}")

    labs["event_time"] = pd.to_datetime(labs[lab_time_col], utc=False, errors="coerce")
    vitals["event_time"] = pd.to_datetime(vitals[vital_time_col], utc=False, errors="coerce")

    merged = pd.merge(
        labs,
        vitals,
        on=[patient_key, "event_time"],
        how="outer",
        suffixes=("_lab", "_vital"),
    )
    merged = merged.sort_values([patient_key, "event_time"])

    adm = admissions.copy()
    if patient_key not in adm.columns:
        raise ValueError(f"admissions missing {patient_key}")
    merged = merged.merge(adm, on=patient_key, how="left", suffixes=("", "_adm"))

    merged = merged.sort_values([patient_key, "event_time"])
    return merged


def timeline_from_demo_longitudinal_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map demo `ehr_data.csv` (patient_id, timestamp, ...) to a MIMIC-like timeline
    with `subject_id` + `charttime` for SQL parity.
    """
    out = df.copy()
    if "patient_id" in out.columns:
        out["subject_id"] = out["patient_id"]
    out["charttime"] = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
    return out.sort_values(["subject_id", "charttime"])
