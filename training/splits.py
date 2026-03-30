"""Train/test splitting with optional patient-level (group) isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def train_test_split_safe(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
):
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Ensure no patient/group appears in both train and test (reduces leakage for longitudinal data).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx = np.arange(len(X))
    train_idx, test_idx = next(gss.split(idx, y, groups))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def temporal_patient_train_test_indices(
    groups: np.ndarray,
    last_event_by_patient: pd.Series,
    *,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split patients by last event time: earliest (1-test_size) fraction → train,
    latest test_size fraction → test. Stronger temporal generalization story than random group split.
    `last_event_by_patient`: index = patient id (same dtype as `groups`), value = datetime-like.
    """
    ordered = last_event_by_patient.sort_values().index.to_numpy()
    n_pat = len(ordered)
    n_test = max(1, int(round(n_pat * test_size)))
    test_patients = set(ordered[-n_test:])
    is_test = np.array([g in test_patients for g in groups], dtype=bool)
    te_i = np.flatnonzero(is_test)
    tr_i = np.flatnonzero(~is_test)
    return tr_i, te_i


def temporal_patient_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    last_event_by_patient: pd.Series,
    *,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tr_i, te_i = temporal_patient_train_test_indices(groups, last_event_by_patient, test_size=test_size)
    return (
        X.iloc[tr_i],
        X.iloc[te_i],
        y.iloc[tr_i],
        y.iloc[te_i],
    )
