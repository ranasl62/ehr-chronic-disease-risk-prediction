import numpy as np
import pandas as pd

from training.splits import group_train_test_split, train_test_split_safe


def test_group_split_no_overlap():
    X = pd.DataFrame({"a": range(20)})
    y = pd.Series([0, 1] * 10)
    groups = np.array([f"p{i // 2}" for i in range(20)])
    X_tr, X_te, y_tr, y_te = group_train_test_split(
        X, y, groups, test_size=0.3, random_state=0
    )
    g_tr = set(groups[X_tr.index.to_numpy()])
    g_te = set(groups[X_te.index.to_numpy()])
    assert g_tr.isdisjoint(g_te)


def test_train_test_split_safe_runs():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0] * 5 + [1] * 5)
    a, b, c, d = train_test_split_safe(X, y, test_size=0.3, random_state=0, stratify=y)
    assert len(a) + len(b) == 10
