"""
SHAP explainers for tree models (XGBoost, Random Forest) and linear pipelines (LR).

Use `scripts/explain_shap.py` to write summary plots to `reports/` (non-interactive).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def _is_xgb_classifier(model: Any) -> bool:
    return type(model).__name__ == "XGBClassifier"


def _is_lgbm_classifier(model: Any) -> bool:
    return type(model).__name__ == "LGBMClassifier"


def _is_tree_shap_model(model: Any) -> bool:
    return (
        _is_xgb_classifier(model)
        or _is_lgbm_classifier(model)
        or isinstance(model, RandomForestClassifier)
    )


def _unwrap_calibrated_estimator(model: Any) -> Any:
    """Use underlying tree/pipeline for SHAP when probabilities are isotonic-calibrated."""
    from sklearn.calibration import CalibratedClassifierCV

    if not isinstance(model, CalibratedClassifierCV):
        return model
    folds = getattr(model, "calibrated_classifiers_", None) or []
    if not folds:
        return model
    first = folds[0]
    return getattr(first, "estimator", first)


def _as_dataframe(X: Any, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X), columns=columns)


def _tree_shap_values(explainer: shap.TreeExplainer, X_explain: pd.DataFrame) -> np.ndarray:
    raw = explainer.shap_values(X_explain)
    if isinstance(raw, list):
        return np.asarray(raw[1])
    return np.asarray(raw)


def explain_model(
    model: Any,
    X_train: Any,
    X_test: Any,
    *,
    plot_path: str | Path | None = None,
    max_background: int = 300,
    max_explain: int = 200,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Global explanation on a subsample of X_test.
    Returns (shap_values_for_positive_class, X_explain_frame).
    """
    X_tr = _as_dataframe(X_train)
    X_te = _as_dataframe(X_test)
    bg_n = min(max_background, len(X_tr))
    ex_n = min(max_explain, len(X_te))
    X_bg = X_tr.sample(n=bg_n, random_state=random_state)
    X_ex = X_te.sample(n=ex_n, random_state=random_state + 1)

    tree_m = _unwrap_calibrated_estimator(model)
    if _is_tree_shap_model(tree_m):
        explainer = shap.TreeExplainer(tree_m)
        shap_values = _tree_shap_values(explainer, X_ex)
    elif isinstance(tree_m, Pipeline) and "scaler" in tree_m.named_steps and "clf" in tree_m.named_steps:
        scaler = tree_m.named_steps["scaler"]
        clf = tree_m.named_steps["clf"]
        X_bg_t = scaler.transform(X_bg)
        X_ex_t = scaler.transform(X_ex)
        explainer = shap.LinearExplainer(clf, X_bg_t)
        shap_values = np.asarray(explainer.shap_values(X_ex_t))
    else:
        raise TypeError(f"Unsupported model for SHAP helper: {type(tree_m)}")

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        if isinstance(tree_m, Pipeline):
            shap.summary_plot(
                shap_values,
                pd.DataFrame(X_ex_t, columns=X_ex.columns),
                show=False,
            )
        else:
            shap.summary_plot(shap_values, X_ex, show=False)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values, X_ex


def explain_single_patient(
    model: Any,
    X_sample: Any,
    *,
    X_background: Any | None = None,
) -> np.ndarray:
    """Local SHAP for one row (DataFrame or 2D array)."""
    xs = _as_dataframe(X_sample)
    if len(xs) != 1:
        raise ValueError("X_sample must contain exactly one row.")

    tree_m = _unwrap_calibrated_estimator(model)
    if _is_tree_shap_model(tree_m):
        explainer = shap.TreeExplainer(tree_m)
        raw = explainer.shap_values(xs)
        if isinstance(raw, list):
            return np.asarray(raw[1]).reshape(1, -1)
        return np.asarray(raw).reshape(1, -1)

    if isinstance(tree_m, Pipeline) and "scaler" in tree_m.named_steps and "clf" in tree_m.named_steps:
        scaler = tree_m.named_steps["scaler"]
        clf = tree_m.named_steps["clf"]
        if X_background is None:
            raise ValueError("X_background required for linear pipeline local SHAP.")
        X_bg = _as_dataframe(X_background)
        X_bg_t = scaler.transform(X_bg)
        x_t = scaler.transform(xs)
        explainer = shap.LinearExplainer(clf, X_bg_t)
        return np.asarray(explainer.shap_values(x_t)).reshape(1, -1)

    raise TypeError(f"Unsupported model for SHAP helper: {type(tree_m)}")
