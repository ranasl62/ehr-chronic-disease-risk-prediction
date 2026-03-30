"""
Subgroup fairness metrics for binary classification (research / reporting).

Requires a discrete group label per row (e.g., age band, sex). Small groups: interpret with caution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def demographic_parity_difference(y_pred: np.ndarray, group: np.ndarray) -> float:
    """Mean absolute difference in positive prediction rate across groups."""
    y_pred = np.asarray(y_pred).astype(int)
    group = np.asarray(group)
    rates = []
    for g in np.unique(group):
        m = group == g
        if m.sum() == 0:
            continue
        rates.append(float(y_pred[m].mean()))
    if len(rates) < 2:
        return float("nan")
    return float(max(rates) - min(rates))


def binary_rates_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
) -> pd.DataFrame:
    """
    Per-group TPR (sensitivity), FPR (1-specificity), prevalence, and counts.
    Requires binary 0/1 labels and predictions.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    group = np.asarray(group)
    rows = []
    for g in np.unique(group):
        m = group == g
        if m.sum() == 0:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        pos = yt == 1
        neg = yt == 0
        tp = int((pos & (yp == 1)).sum())
        fn = int((pos & (yp == 0)).sum())
        fp = int((neg & (yp == 1)).sum())
        tn = int((neg & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        rows.append(
            {
                "group": g,
                "n": int(m.sum()),
                "prevalence": float(yt.mean()),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group: np.ndarray,
) -> pd.DataFrame:
    """Per-group count, prevalence, accuracy, and mean predicted probability."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    group = np.asarray(group)
    rows = []
    for g in np.unique(group):
        m = group == g
        if m.sum() == 0:
            continue
        yt, yp, pr = y_true[m], y_pred[m], y_prob[m]
        rows.append(
            {
                "group": g,
                "n": int(m.sum()),
                "prevalence": float(yt.mean()),
                "accuracy": float((yt == yp).mean()),
                "mean_predicted_prob": float(pr.mean()),
            }
        )
    return pd.DataFrame(rows)
