import numpy as np
import pandas as pd

from fairness.bias_metrics import (
    binary_rates_by_group,
    demographic_parity_difference,
    subgroup_metrics_table,
)


def test_binary_rates_by_group():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    g = np.array(["A", "A", "A", "B", "B", "B"])
    df = binary_rates_by_group(y_true, y_pred, g)
    assert "tpr" in df.columns and "fpr" in df.columns
    assert len(df) == 2


def test_subgroup_table():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.6, 0.7])
    g = np.array(["A", "A", "B", "B"])
    df = subgroup_metrics_table(y_true, y_pred, y_prob, g)
    assert len(df) == 2
    assert "n" in df.columns


def test_dpd():
    y_pred = np.array([1, 0, 1, 0])
    g = np.array(["A", "A", "B", "B"])
    d = demographic_parity_difference(y_pred, g)
    assert abs(d) < 1e-9
