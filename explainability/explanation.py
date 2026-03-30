"""
Patient-level explanation JSON (SHAP when available; structured fallback).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from explainability.shap_explainer import explain_single_patient


def _top_k_pairs(names: list[str], values: np.ndarray, k: int = 12) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(values))
    out: list[dict[str, Any]] = []
    for i in order[:k]:
        j = int(i)
        out.append(
            {
                "feature": names[j],
                "shap_value": float(values[j]),
                "abs_contribution": float(abs(values[j])),
            }
        )
    return out


def build_patient_explanation(
    artifact: dict[str, Any],
    features: dict[str, float],
    *,
    top_k: int = 12,
    X_background: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Returns JSON-serializable explanation for one patient feature vector.
    """
    cols: list[str] = artifact["feature_columns"]
    row = pd.DataFrame([{c: float(features[c]) for c in cols}])
    model = artifact["model"]
    bg = X_background if X_background is not None else artifact.get("shap_background")

    expl: dict[str, Any] = {
        "method": "shap",
        "top_positive_risk_drivers": [],
        "notes": [],
    }

    try:
        sv = explain_single_patient(model, row, X_background=bg)
        vec = np.asarray(sv).ravel()
        if len(vec) != len(cols):
            raise ValueError("SHAP length mismatch")
        expl["top_positive_risk_drivers"] = _top_k_pairs(cols, vec, k=top_k)
        expl["shap_vector"] = {c: float(v) for c, v in zip(cols, vec, strict=True)}
        return expl
    except Exception as e:
        expl["method"] = "fallback_global_importance"
        expl["notes"].append(f"SHAP unavailable ({e!s}); using artifact global importance ranking.")
        fi = artifact.get("feature_importance") or {}
        if not fi:
            expl["notes"].append("No feature_importance in artifact; run training.reporting.")
            return expl
        pairs = sorted(fi.items(), key=lambda x: -abs(x[1]))[:top_k]
        expl["top_positive_risk_drivers"] = [
            {"feature": n, "global_importance": float(v), "abs_contribution": float(abs(v))}
            for n, v in pairs
        ]
        return expl


def merge_prediction_and_explanation(
    pred: dict[str, Any],
    explanation: dict[str, Any],
) -> dict[str, Any]:
    return {"prediction": pred, "explanation": explanation}
