from pathlib import Path
from typing import Any

import joblib
import numpy as np

from inference.validation import validate_feature_dict
from utils.config import MODEL_PATH


def risk_level_from_probability(
    p: float,
    high: float = 0.7,
    mid: float = 0.4,
) -> str:
    if p > high:
        return "high"
    if p > mid:
        return "medium"
    return "low"


def load_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Missing model artifact: {p}. Run: python -m training.train")
    art: dict[str, Any] = joblib.load(p)
    fc = art.get("feature_columns")
    if fc is not None:
        art["feature_columns"] = [str(c) for c in list(fc)]
    return art


def predict_row(features: dict[str, float], artifact: dict | None = None) -> dict[str, Any]:
    art = artifact or load_artifact()
    model = art["model"]
    cols: list[str] = art["feature_columns"]
    validate_feature_dict(features, required_columns=cols, allow_extra=False)
    row = np.array([[float(features[c]) for c in cols]])
    proba = float(model.predict_proba(row)[0, 1])
    cls = int(model.predict(row)[0])
    return {
        "risk_probability": proba,
        "risk_class": cls,
        "risk_score": proba,
        "risk_level": risk_level_from_probability(proba),
    }
