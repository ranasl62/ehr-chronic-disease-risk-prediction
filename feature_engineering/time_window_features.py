"""
Time-window aggregation per patient anchored at an index time.

Uses only rows in [index_time - window_days, index_time] (or half-open upper bound).
Pass an explicit index_times Series for cohort studies; otherwise defaults to last event.
"""

from __future__ import annotations

import pandas as pd

from preprocessing.time_windowing import filter_events_to_window


def create_time_window_features(
    df: pd.DataFrame,
    window_days: int = 180,
    time_col: str = "timestamp",
    *,
    index_times: pd.Series | None = None,
    inclusive: bool = True,
) -> pd.DataFrame:
    df = df.sort_values(["patient_id", time_col])
    features: list[dict] = []

    for pid, group in df.groupby("patient_id", sort=False):
        group = group.sort_values(time_col)
        if index_times is not None and pid in index_times.index:
            index_time = pd.Timestamp(index_times.loc[pid])
        else:
            index_time = group[time_col].max()
        recent = filter_events_to_window(
            group,
            index_time,
            window_days,
            time_col=time_col,
            inclusive=inclusive,
        )
        if recent.empty:
            # Fall back to last pre-index row if any; else skip empty patient.
            pre = group[group[time_col] <= index_time] if inclusive else group[group[time_col] < index_time]
            if pre.empty:
                continue
            recent = pre.iloc[[-1]]

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
