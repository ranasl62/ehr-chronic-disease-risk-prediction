"""Expected calibration error (ECE) and related diagnostics for binary risk scores."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Mean absolute gap between predicted probability and empirical prevalence per bin.
    Lower is better (0 = perfectly calibrated on this discretization).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        return float("nan")
    if strategy != "uniform":
        raise ValueError("Only strategy='uniform' is supported.")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not m.any():
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        w = float(m.sum()) / n
        ece += w * abs(acc - conf)
    return float(ece)
