"""
Rebuild (X, y, groups) and the same train/test row indices as `training.train.run_training`.

Used by SHAP scripts, fairness reports, and leakage audits so holdout work matches training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from preprocessing.cleaning import clean_longitudinal_ehr
from preprocessing.ehr_loader import load_data, load_ehr_data
from training.splits import temporal_patient_train_test_indices
from training.train import build_xy_longitudinal, build_xy_tabular


def load_xy_groups_from_artifact(artifact: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    fe = artifact.get("feature_engineering") or {}
    fmt = fe.get("format", "tabular")
    data_path = Path(fe.get("data_path", ""))
    if not data_path.is_file():
        raise FileNotFoundError(f"Artifact data_path not found: {data_path}")

    if fmt == "longitudinal":
        df = load_ehr_data(data_path)
        wd_list = fe.get("windows_days")
        if wd_list:
            X, y, _, groups = build_xy_longitudinal(df, windows_days=tuple(wd_list))
        else:
            X, y, _, groups = build_xy_longitudinal(
                df,
                window_days=int(fe.get("window_days") or 180),
            )
    else:
        df = load_data(data_path)
        X, y, _, groups = build_xy_tabular(df)
    return X, y, np.asarray(groups)


def reproducible_train_test_indices(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    *,
    split_method: str,
    test_size: float,
    random_state: int,
    fe: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) integer positions aligned with X.iloc."""
    idx = np.arange(len(X))
    if split_method == "temporal_patient":
        if not fe:
            raise ValueError("temporal_patient split requires feature_engineering meta (fe).")
        dp = Path(fe.get("data_path", ""))
        if not dp.is_file():
            raise FileNotFoundError(f"temporal_patient: data_path missing: {dp}")
        df = clean_longitudinal_ehr(load_ehr_data(dp))
        last_e = df.groupby("patient_id")["timestamp"].max()
        return temporal_patient_train_test_indices(groups, last_e, test_size=test_size)
    if split_method == "patient_group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(idx, y, groups))
        return train_idx, test_idx
    try:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
        )
    return train_idx, test_idx


def split_train_test_from_artifact(
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Same matrices and indices as training used (per feature_engineering meta).
    """
    fe = artifact.get("feature_engineering") or {}
    X, y, groups = load_xy_groups_from_artifact(artifact)
    split_method = fe.get("split_method", "random_row")
    test_size = float(fe.get("test_size", 0.2))
    random_state = int(fe.get("random_state", 42))
    tr_i, te_i = reproducible_train_test_indices(
        X,
        y,
        groups,
        split_method=split_method,
        test_size=test_size,
        random_state=random_state,
        fe=fe,
    )
    return (
        X.iloc[tr_i],
        X.iloc[te_i],
        y.iloc[tr_i],
        y.iloc[te_i],
        tr_i,
        te_i,
    )
