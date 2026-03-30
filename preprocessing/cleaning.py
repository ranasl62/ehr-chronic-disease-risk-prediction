import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["patient_id"]).copy()
    numeric = ["age", "glucose", "blood_pressure", "cholesterol"]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_longitudinal_ehr(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types for MIMIC-style longitudinal tables."""
    out = df.dropna(subset=["patient_id", "timestamp"]).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    for col in [
        "age",
        "glucose",
        "blood_pressure",
        "cholesterol",
        "lab_value",
        "vital_signs",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "icd_code" in out.columns:
        out["icd_code"] = out["icd_code"].astype(str)
    return out
