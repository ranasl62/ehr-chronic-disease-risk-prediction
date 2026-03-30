"""
Time-window aggregation per patient anchored at last observed timestamp.

Uses only rows in [last_time - window_days, last_time]. For production cohorts,
truncate each patient to prediction-time index before calling this helper to
avoid label leakage.
"""

import pandas as pd

from preprocessing.time_windowing import filter_events_to_window


def create_time_window_features(
    df: pd.DataFrame,
    window_days: int = 180,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    df = df.sort_values(["patient_id", time_col])
    features: list[dict] = []

    for pid, group in df.groupby("patient_id", sort=False):
        group = group.sort_values(time_col)
        index_time = group[time_col].max()
        recent = filter_events_to_window(group, index_time, window_days, time_col=time_col)
        if recent.empty:
            recent = group.iloc[[-1]]

        row: dict = {"patient_id": pid, "visit_count": float(len(recent))}
        if "glucose" in recent.columns:
            row["glucose"] = float(recent["glucose"].mean())
        if "blood_pressure" in recent.columns:
            row["blood_pressure"] = float(recent["blood_pressure"].mean())
        if "cholesterol" in recent.columns:
            row["cholesterol"] = float(recent["cholesterol"].mean())
        if "age" in recent.columns:
            row["age"] = float(recent["age"].iloc[-1])
        if "lab_value" in recent.columns:
            row["lab_value_mean"] = float(recent["lab_value"].mean())
        if "vital_signs" in recent.columns:
            row["vital_signs_mean"] = float(recent["vital_signs"].mean())
        if "icd_code" in recent.columns:
            row["icd_unique_count"] = float(recent["icd_code"].nunique())

        features.append(row)

    return pd.DataFrame(features)
