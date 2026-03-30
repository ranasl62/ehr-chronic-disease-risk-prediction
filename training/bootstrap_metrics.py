"""Bootstrap confidence intervals for hold-out metrics (research reporting)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_roc_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | None]:
    """
    Percentile bootstrap on test rows. Returns None for stats if AUC undefined (single class).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return {"roc_auc_point": float("nan"), "roc_auc_ci_low": None, "roc_auc_ci_high": None, "n_bootstrap": n_bootstrap}
    point = float(roc_auc_score(y_true, y_prob))
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, yp))
    if not scores:
        return {"roc_auc_point": point, "roc_auc_ci_low": None, "roc_auc_ci_high": None, "n_bootstrap": n_bootstrap}
    alpha = (1.0 - confidence) / 2
    lo, hi = np.quantile(scores, [alpha, 1.0 - alpha])
    return {
        "roc_auc_point": point,
        "roc_auc_ci_low": float(lo),
        "roc_auc_ci_high": float(hi),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence,
    }
