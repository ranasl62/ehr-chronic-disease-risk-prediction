"""Shared validation for prediction requests (API, dashboard, batch)."""

from __future__ import annotations

import math
from typing import Any


def validate_feature_dict(
    features: dict[str, float],
    *,
    required_columns: list[str],
    allow_extra: bool = False,
) -> None:
    """
    Ensure keys match and all required values are finite floats.
    Raises ValueError with messages suitable for API 400 responses.
    """
    if not allow_extra:
        extra = [k for k in features if k not in required_columns]
        if extra:
            raise ValueError(
                f"Unexpected feature keys {extra}; expected exactly {list(required_columns)}."
            )
    missing = [c for c in required_columns if c not in features]
    if missing:
        raise ValueError(f"Missing feature keys {missing}.")
    for c in required_columns:
        v = features[c]
        try:
            x = float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Feature {c!r} must be numeric.") from e
        if not math.isfinite(x):
            raise ValueError(f"Feature {c!r} must be a finite number (not NaN or Inf).")


def build_input_stats_frame(X_train) -> dict[str, dict[str, float]]:
    """Per-feature training distribution summaries for UI defaults and API schema."""
    import pandas as pd

    out: dict[str, dict[str, float]] = {}
    for c in X_train.columns:
        s = pd.to_numeric(X_train[c], errors="coerce").dropna()
        if len(s) == 0:
            out[str(c)] = {"median": 0.0, "p05": 0.0, "p95": 0.0, "mean": 0.0}
            continue
        out[str(c)] = {
            "median": float(s.median()),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
            "mean": float(s.mean()),
        }
    return out
