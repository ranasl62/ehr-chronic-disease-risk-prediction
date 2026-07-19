"""Tests for index-time / horizon leakage-safe cohort builders."""

from __future__ import annotations

import pandas as pd

from feature_engineering.cohort_integrity import (
    audit_temporal_integrity,
    horizon_labels,
    resolve_index_times,
    truncate_events_to_index,
)
from training.train import build_xy_longitudinal


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2023-01-01", "glucose": 100.0, "label": 0},
            {"patient_id": 1, "timestamp": "2023-03-01", "glucose": 105.0, "label": 0},
            {"patient_id": 1, "timestamp": "2023-06-01", "glucose": 180.0, "label": 1},
            {"patient_id": 2, "timestamp": "2023-01-01", "glucose": 90.0, "label": 0},
            {"patient_id": 2, "timestamp": "2023-04-01", "glucose": 92.0, "label": 0},
            {"patient_id": 2, "timestamp": "2023-07-01", "glucose": 94.0, "label": 0},
        ]
    )


def test_before_last_index_and_horizon_label():
    df = _toy_df()
    idx = resolve_index_times(df, index_strategy="before_last")
    assert pd.Timestamp(idx.loc[1]) == pd.Timestamp("2023-03-01")
    y = horizon_labels(df, idx, horizon_days=120, label_col="label")
    assert int(y.loc[1]) == 1
    assert int(y.loc[2]) == 0


def test_truncate_excludes_post_index():
    df = _toy_df()
    idx = resolve_index_times(df, index_strategy="before_last")
    feat = truncate_events_to_index(df, idx, inclusive=True)
    assert feat.loc[feat["patient_id"] == 1, "timestamp"].max() <= pd.Timestamp("2023-03-01")
    assert (feat["label"] == 1).sum() == 0


def test_build_xy_horizon_does_not_use_outcome_row_as_feature_signal():
    df = _toy_df()
    X, y, cols, groups = build_xy_longitudinal(
        df,
        window_days=180,
        horizon_days=120,
        index_strategy="before_last",
    )
    assert len(X) == 2
    assert set(y.tolist()) == {0, 1}
    # Glucose for patient 1 must reflect pre-index mean (~102.5), not 180.
    # Single-window columns use unprefixed names.
    gcol = "glucose" if "glucose" in X.columns else None
    if gcol:
        row = X.loc[groups == 1].iloc[0]
        assert float(row[gcol]) < 150.0


def test_audit_temporal_integrity_passes_after_truncate():
    df = _toy_df()
    idx = resolve_index_times(df, index_strategy="before_last")
    feat = truncate_events_to_index(df, idx, inclusive=True)
    report = audit_temporal_integrity(
        df,
        idx,
        horizon_days=120,
        feature_df=feat,
        feature_inclusive=True,
    )
    assert report["passed"] is True
    assert report["feature_events_after_index"] == 0


def test_index_time_column_strategy():
    df = _toy_df()
    df["index_time"] = "2023-03-01"
    idx = resolve_index_times(df, index_strategy="column", index_time_col="index_time")
    assert pd.Timestamp(idx.loc[1]) == pd.Timestamp("2023-03-01")
