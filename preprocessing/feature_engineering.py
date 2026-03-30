"""
EHR feature engineering orchestration (temporal, multi-window).

**Data lineage (longitudinal demo → production MIMIC-IV)**

1. Raw: event-level rows (`patient_id`/`subject_id`, `timestamp`/`charttime`, labs, vitals, ICD).
2. `preprocessing.cleaning.clean_longitudinal_ehr` — type coercion, drop null keys.
3. `feature_engineering.multi_window.merge_multi_window_features` — aggregates in **7d, 30d, 180d**
   lookbacks anchored at **last observed time** per patient (replace anchor with cohort index_time).
4. Labels merged at patient level; features must use only information **≤ anchor** to avoid leakage.

For SQL extraction patterns see `sql/mimic_queries.sql`.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.multi_window import merge_multi_window_features
from feature_engineering.patient_features import create_features
from feature_engineering.time_window_features import create_time_window_features

__all__ = [
    "create_features",
    "create_time_window_features",
    "merge_multi_window_features",
    "build_tabular_patient_matrix",
    "build_longitudinal_multiwindow_matrix",
]


def build_tabular_patient_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient aggregates from wide/tabular-style encounters (demo CSV)."""
    return create_features(df)


def build_longitudinal_multiwindow_matrix(
    df: pd.DataFrame,
    *,
    windows_days: tuple[int, ...] = (7, 30, 180),
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Patient-level matrix with `w7d_*`, `w30d_*`, `w180d_*` columns."""
    return merge_multi_window_features(
        df,
        windows_days=windows_days,
        patient_col=patient_col,
        time_col=time_col,
    )
