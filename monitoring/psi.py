"""
Population Stability Index (PSI) — simple drift diagnostic between two probability or score distributions.

Common rule of thumb: PSI < 0.1 stable, 0.1–0.25 watch, > 0.25 investigate.
"""

from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10, eps: float = 1e-6) -> float:
    e = np.asarray(expected, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    if len(e) < 2 or len(a) < 2:
        return float("nan")
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(e, qs))
    if len(bins) < 2:
        return 0.0
    e_hist, _ = np.histogram(e, bins=bins)
    a_hist, _ = np.histogram(a, bins=bins)
    e_pct = (e_hist + eps) / (e_hist.sum() + eps * len(e_hist))
    a_pct = (a_hist + eps) / (a_hist.sum() + eps * len(a_hist))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
