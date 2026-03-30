"""
Persist evaluation metrics and global feature importance for reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate import evaluate_binary


def feature_importance_from_model(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Model-agnostic global importance (approximation for linear / tree ensembles)."""
    names = list(feature_names)
    m = model

    from sklearn.calibration import CalibratedClassifierCV

    if isinstance(m, CalibratedClassifierCV) and m.calibrated_classifiers_:
        m = m.calibrated_classifiers_[0].estimator

    from sklearn.pipeline import Pipeline

    if isinstance(m, Pipeline) and "clf" in m.named_steps:
        clf = m.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_).ravel()
            if len(coef) == len(names):
                return {n: float(abs(c)) for n, c in zip(names, coef, strict=True)}
        if hasattr(clf, "feature_importances_"):
            imp = np.asarray(clf.feature_importances_)
            if len(imp) == len(names):
                return {n: float(v) for n, v in zip(names, imp, strict=True)}
        return {}

    if hasattr(m, "feature_importances_"):
        imp = np.asarray(m.feature_importances_)
        if len(imp) == len(names):
            return {n: float(v) for n, v in zip(names, imp, strict=True)}

    if hasattr(m, "coef_"):
        coef = np.asarray(m.coef_).ravel()
        if len(coef) == len(names):
            return {n: float(abs(c)) for n, c in zip(names, coef, strict=True)}

    return {n: 0.0 for n in names}


def build_evaluation_report(
    model: Any,
    X_test,
    y_test,
    *,
    meta: dict[str, Any] | None = None,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, Any]:
    m = evaluate_binary(model, X_test, y_test, threshold=threshold, ece_bins=ece_bins)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "metrics": {k: v for k, v in m.items() if k != "report"},
        "classification_report_text": m["report"],
        "meta": meta or {},
    }
    return report


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
