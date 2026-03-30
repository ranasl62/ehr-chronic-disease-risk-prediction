import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.calibration_metrics import expected_calibration_error


def evaluate_binary(model, X_test, y_test, threshold: float = 0.5, *, ece_bins: int = 10):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_test_arr = np.asarray(y_test)
    single_class = len(np.unique(y_test_arr)) < 2
    if single_class:
        roc = float("nan")
        pr_auc = float("nan")
    else:
        roc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    ece = expected_calibration_error(y_test_arr, y_prob, n_bins=ece_bins)
    ece_f = float(ece) if np.isfinite(ece) else float("nan")
    out = {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(y_test, y_prob)),
        "ece": ece_f,
        "ece_bins": ece_bins,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return out


def print_metrics(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
    *,
    ece_bins: int = 10,
) -> None:
    m = evaluate_binary(model, X_test, y_test, threshold=threshold, ece_bins=ece_bins)
    print("--- Hold-out evaluation (research-style summary) ---")
    if np.isnan(m["roc_auc"]):
        print("ROC-AUC (AUC-ROC):     n/a (single class in holdout)")
    else:
        print(f"ROC-AUC (AUC-ROC):     {m['roc_auc']:.4f}")
    if np.isnan(m["pr_auc"]):
        print("PR-AUC (avg prec.):    n/a (single class in holdout)")
    else:
        print(f"PR-AUC (avg prec.):    {m['pr_auc']:.4f}")
    print(f"Brier score:           {m['brier']:.4f}  (lower is better; related to calibration)")
    _ece = m.get("ece", float("nan"))
    if np.isfinite(_ece):
        print(f"ECE ({m.get('ece_bins', 10)} bins):          {_ece:.4f}  (lower is better)")
    print(f"Accuracy @ {threshold}: {m['accuracy']:.4f}")
    print(f"Precision @ {threshold}: {m['precision']:.4f}")
    print(f"Recall @ {threshold}:    {m['recall']:.4f}")
    print(f"F1 @ {threshold}:        {m['f1']:.4f}")
    print("Classification report:")
    print(m["report"])


def print_lead_time_summary(
    lead_days,
    y_true,
    y_prob,
    *,
    threshold: float = 0.5,
) -> None:
    """
    Lead-time gain: among high-risk predictions that are true positives,
    summarize days from prediction time to diagnosis (or outcome) time.

    `lead_days` must align row-wise with y_true / y_prob.
    """
    yt = np.asarray(y_true).astype(int)
    pr = np.asarray(y_prob).astype(float)
    high = pr >= threshold
    tp = high & (yt == 1)
    print("--- Lead-time gain (research metric) ---")
    if not tp.any():
        print("No true-positive high-risk predictions at this threshold; lead-time n/a.")
        return
    ld = np.asarray(lead_days, dtype=float)
    vals = pd.Series(ld[tp]).dropna()
    if vals.empty:
        print("Lead-time values missing for true-positive high-risk rows.")
        return
    print(
        f"Among TP @ prob≥{threshold}: median_days={vals.median():.1f}, "
        f"mean_days={vals.mean():.1f}, n={len(vals)}"
    )
