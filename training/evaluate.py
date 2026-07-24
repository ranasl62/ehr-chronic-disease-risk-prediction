import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from training.calibration_metrics import calibration_curve_points, expected_calibration_error


def _finite_list(arr, *, max_points: int = 200) -> list[float]:
    """Downsample curve arrays for JSON/UI without changing shape much."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return []
    if a.size > max_points:
        idx = np.linspace(0, a.size - 1, max_points).astype(int)
        a = a[idx]
    out: list[float] = []
    for v in a:
        if np.isfinite(v):
            out.append(float(v))
    return out


def curve_payload(y_true, y_prob, *, ece_bins: int = 10) -> dict:
    """
    ROC / PR / calibration points for research Chart.js plots.
    Single-class holdouts return empty curves plus an explicit note.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()
    notes: list[str] = []
    roc = {"fpr": [], "tpr": [], "thresholds": []}
    pr = {"precision": [], "recall": [], "thresholds": []}
    if len(y_true_arr) == 0 or len(y_true_arr) != len(y_prob_arr):
        notes.append("curves_unavailable_empty_or_mismatched")
        return {
            "roc": roc,
            "pr": pr,
            "calibration": calibration_curve_points(y_true_arr, y_prob_arr, n_bins=ece_bins),
            "notes": notes,
        }
    if len(np.unique(y_true_arr)) < 2:
        notes.append("single_class_holdout_curves_unavailable")
    else:
        fpr, tpr, thr_roc = roc_curve(y_true_arr, y_prob_arr)
        prec, rec, thr_pr = precision_recall_curve(y_true_arr, y_prob_arr)
        roc = {
            "fpr": _finite_list(fpr),
            "tpr": _finite_list(tpr),
            "thresholds": _finite_list(thr_roc),
        }
        pr = {
            "precision": _finite_list(prec),
            "recall": _finite_list(rec),
            # precision_recall_curve thresholds are len-1 vs points
            "thresholds": _finite_list(thr_pr),
        }
    cal = calibration_curve_points(y_true_arr, y_prob_arr, n_bins=ece_bins)
    if not cal["counts"]:
        notes.append("calibration_bins_empty")
    return {"roc": roc, "pr": pr, "calibration": cal, "notes": notes}


def bootstrap_metric_cis(
    y_true,
    y_prob,
    *,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Percentile bootstrap CIs for ROC-AUC / PR-AUC on the hold-out set.
    Returns null intervals when the hold-out is single-class or too small.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()
    empty = {
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "roc_auc_ci": None,
        "pr_auc_ci": None,
        "note": "unavailable",
    }
    n = len(y_true_arr)
    if n < 8 or len(y_true_arr) != len(y_prob_arr) or len(np.unique(y_true_arr)) < 2:
        empty["note"] = "holdout_too_small_or_single_class"
        return empty
    rng = np.random.default_rng(seed)
    roc_s: list[float] = []
    pr_s: list[float] = []
    for _ in range(max(10, int(n_boot))):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_prob_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            roc_s.append(float(roc_auc_score(yt, yp)))
            pr_s.append(float(average_precision_score(yt, yp)))
        except ValueError:
            continue
    if len(roc_s) < 10:
        empty["note"] = "insufficient_valid_bootstrap_replicates"
        return empty
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "n_valid": len(roc_s),
        "roc_auc_ci": [float(np.percentile(roc_s, lo_q)), float(np.percentile(roc_s, hi_q))],
        "pr_auc_ci": [float(np.percentile(pr_s, lo_q)), float(np.percentile(pr_s, hi_q))],
        "note": "percentile_bootstrap",
    }


def evaluate_binary(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
    *,
    ece_bins: int = 10,
    include_curves: bool = False,
    bootstrap_cis: bool = False,
    n_boot: int = 200,
):
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
        "n_holdout": int(len(y_test_arr)),
        "single_class_holdout": bool(single_class),
    }
    if include_curves:
        out["curves"] = curve_payload(y_test_arr, y_prob, ece_bins=ece_bins)
    if bootstrap_cis:
        out["bootstrap_cis"] = bootstrap_metric_cis(
            y_test_arr, y_prob, n_boot=n_boot, seed=42
        )
    return out


def threshold_operating_points(
    y_true,
    y_prob,
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    """Precision/recall/F1 at a few decision thresholds (research operating-point table)."""
    thr_list = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, float]] = []
    for thr in thr_list:
        y_pred = (y_prob_arr >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true_arr, y_pred)),
                "positive_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
            }
        )
    return rows


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
