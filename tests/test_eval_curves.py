"""Curve payloads + bootstrap CIs for research evaluation plots."""

from __future__ import annotations

import numpy as np

from training.calibration_metrics import calibration_curve_points
from training.evaluate import bootstrap_metric_cis, curve_payload
from training.reporting import build_evaluation_report


class _ProbModel:
    def __init__(self, probs: np.ndarray):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        p = self._probs[:n]
        return np.column_stack([1.0 - p, p])


def test_curve_payload_two_class():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.6, 0.4])
    curves = curve_payload(y, p, ece_bins=5)
    assert curves["roc"]["fpr"]
    assert curves["pr"]["precision"]
    assert curves["calibration"]["counts"]
    assert "single_class" not in "".join(curves["notes"])


def test_curve_payload_single_class_notes():
    y = np.zeros(10, dtype=int)
    p = np.linspace(0.1, 0.9, 10)
    curves = curve_payload(y, p)
    assert curves["roc"]["fpr"] == []
    assert any("single_class" in n for n in curves["notes"])


def test_calibration_curve_points_empty():
    empty = calibration_curve_points(np.array([]), np.array([]))
    assert empty["counts"] == []


def test_bootstrap_cis_small_holdout():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    out = bootstrap_metric_cis(y, p, n_boot=20)
    assert out["roc_auc_ci"] is None


def test_bootstrap_cis_ok():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=40)
    p = np.clip(y * 0.6 + rng.random(40) * 0.4, 0, 1)
    out = bootstrap_metric_cis(y, p, n_boot=50, seed=1)
    assert out["roc_auc_ci"] is not None
    lo, hi = out["roc_auc_ci"]
    assert 0.0 <= lo <= hi <= 1.0


def test_build_evaluation_report_includes_curves(tmp_path):
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.85, 0.75, 0.3, 0.9, 0.65, 0.35, 0.8, 0.25, 0.7, 0.4, 0.88, 0.6, 0.15, 0.45])
    X = np.zeros((len(y), 1))
    model = _ProbModel(p)
    report = build_evaluation_report(model, X, y, n_boot=40)
    assert "curves" in report
    assert report["curves"]["roc"]["tpr"]
    assert "bootstrap_cis" in report
    assert report.get("quality_note")
