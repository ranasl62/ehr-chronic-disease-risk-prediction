"""
Multi lookback windows (e.g. 7d / 30d / 180d) with strict temporal bounds.

Each window is anchored at the patient's last observed time and only includes
rows with t in [anchor - W, anchor]. For cohort studies, truncate events to
strictly before clinical index_time before calling these builders.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.time_window_features import create_time_window_features


def merge_multi_window_features(
    df: pd.DataFrame,
    windows_days: tuple[int, ...] = (7, 30, 180),
    *,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    One row per patient with columns prefixed `w{W}d_` per window (except patient_id).
    """
    merged: pd.DataFrame | None = None
    for w in windows_days:
        block = create_time_window_features(df, window_days=w, time_col=time_col)
        rename = {
            c: f"w{w}d_{c}"
            for c in block.columns
            if c != patient_col
        }
        block = block.rename(columns=rename)
        if merged is None:
            merged = block
        else:
            merged = merged.merge(block, on=patient_col, how="outer")
    if merged is None:
        return pd.DataFrame()
    return merged
