"""
Index-time and prediction-horizon helpers for leakage-safe longitudinal cohorts.

Features use events with timestamp <= index (inclusive) or < index (exclusive).
Labels use only events strictly after index within horizon H days.
"""

from __future__ import annotations

import pandas as pd


def resolve_index_times(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    index_time_col: str | None = None,
    index_strategy: str = "last_event",
) -> pd.Series:
    """
    Return a Series indexed by patient_id with one Timestamp index per patient.

    Strategies:
    - ``column``: require ``index_time_col`` on every row (first non-null per patient).
    - ``last_event``: max(timestamp) — legacy demo default; pair with horizon only if
      post-index outcome rows exist.
    - ``before_last``: second-to-last distinct timestamp when ≥2 events; else last.
      Useful for demo CSVs where the final row carries the outcome signal.
    """
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    if index_strategy == "column":
        col = index_time_col or "index_time"
        if col not in work.columns:
            raise ValueError(f"index_strategy='column' requires column {col!r}")
        work[col] = pd.to_datetime(work[col], errors="coerce")
        idx = work.groupby(patient_col, sort=False)[col].first()
        if idx.isna().any():
            raise ValueError(f"Missing {col} for some patients.")
        return idx

    if index_strategy == "before_last":
        out: dict = {}
        for pid, g in work.groupby(patient_col, sort=False):
            times = g[time_col].dropna().sort_values().unique()
            if len(times) >= 2:
                out[pid] = pd.Timestamp(times[-2])
            elif len(times) == 1:
                out[pid] = pd.Timestamp(times[0])
            else:
                raise ValueError(f"Patient {pid} has no valid timestamps.")
        return pd.Series(out, name="index_time")

    if index_strategy != "last_event":
        raise ValueError(f"Unknown index_strategy: {index_strategy!r}")
    idx = work.groupby(patient_col, sort=False)[time_col].max()
    if idx.isna().any():
        raise ValueError("Some patients have no valid timestamps for last_event index.")
    return idx


def truncate_events_to_index(
    df: pd.DataFrame,
    index_times: pd.Series,
    *,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    inclusive: bool = True,
) -> pd.DataFrame:
    """Keep only pre-index (and optionally at-index) events for feature building."""
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    mapped = work[patient_col].map(index_times)
    if inclusive:
        mask = work[time_col] <= mapped
    else:
        mask = work[time_col] < mapped
    return work.loc[mask].copy()


def horizon_labels(
    df: pd.DataFrame,
    index_times: pd.Series,
    *,
    horizon_days: int,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    label_col: str = "label",
) -> pd.Series:
    """
    Binary label per patient: any positive ``label_col`` in (index, index + H].

    Patients with no post-index events in the horizon receive label 0.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    if label_col not in work.columns:
        raise ValueError(f"Label column not found: {label_col!r}")

    labels: dict = {}
    for pid, t_index in index_times.items():
        t_end = pd.Timestamp(t_index) + pd.Timedelta(days=horizon_days)
        g = work.loc[work[patient_col] == pid]
        post = g[(g[time_col] > t_index) & (g[time_col] <= t_end)]
        if post.empty:
            labels[pid] = 0
        else:
            labels[pid] = int(post[label_col].fillna(0).astype(float).max() >= 1)
    return pd.Series(labels, name=label_col)


def audit_temporal_integrity(
    df: pd.DataFrame,
    index_times: pd.Series,
    *,
    horizon_days: int | None,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    label_col: str = "label",
    feature_inclusive: bool = True,
    feature_df: pd.DataFrame | None = None,
) -> dict:
    """
    Check that feature rows do not use post-index times.
    """
    feat_src = feature_df if feature_df is not None else df
    feat_src = feat_src.copy()
    feat_src[time_col] = pd.to_datetime(feat_src[time_col], errors="coerce")

    n_future_feat = 0
    for pid, t_index in index_times.items():
        g = feat_src.loc[feat_src[patient_col] == pid]
        if feature_inclusive:
            bad = g[g[time_col] > t_index]
        else:
            bad = g[g[time_col] >= t_index]
        n_future_feat += int(len(bad))

    passed = n_future_feat == 0
    notes: list[str] = []
    if not passed:
        notes.append("CRITICAL: feature events found after index_time.")
    if horizon_days is not None:
        notes.append(
            f"Horizon labels use events in (index, index+{horizon_days}d] only."
        )
    return {
        "feature_events_after_index": int(n_future_feat),
        "feature_inclusive": bool(feature_inclusive),
        "horizon_days": horizon_days,
        "n_patients": int(len(index_times)),
        "passed": passed,
        "notes": notes,
        "label_col": label_col,
    }
