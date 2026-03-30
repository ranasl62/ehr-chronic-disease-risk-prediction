from pathlib import Path

import pandas as pd

from utils.config import DEFAULT_EHR_LONGITUDINAL_CSV


def load_data(path: str | Path) -> pd.DataFrame:
    """Load any supported CSV (tabular or raw path)."""
    return pd.read_csv(path)


def load_ehr_data(path: str | Path | None = None) -> pd.DataFrame:
    """
    Longitudinal EHR loader (MIMIC-style column expectations).

    Expected columns (superset; extras ignored if unused):
        patient_id, timestamp, icd_code, lab_value, vital_signs,
        glucose, blood_pressure, cholesterol, age, label (or chronic_disease)

    Replace path with real MIMIC-IV extracts when credentialed.
    For heterogeneous column names (e.g. claims exports), use
    `preprocessing.canonical_schema.rename_to_canonical_longitudinal` or
    `scripts/normalize_longitudinal_csv.py`.
    """
    p = Path(path or DEFAULT_EHR_LONGITUDINAL_CSV)
    df = pd.read_csv(p)
    if "timestamp" not in df.columns:
        raise ValueError("Longitudinal EHR CSV must include a 'timestamp' column.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    return df
