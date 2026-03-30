import numpy as np
import pandas as pd

from training.splits import temporal_patient_train_test_indices


def test_temporal_split_disjoint_and_orders_late_to_test():
    groups = np.array([1, 1, 2, 2, 3, 3])
    last_e = pd.Series(
        {
            1: pd.Timestamp("2020-01-01"),
            2: pd.Timestamp("2020-06-01"),
            3: pd.Timestamp("2020-12-01"),
        }
    )
    tr_i, te_i = temporal_patient_train_test_indices(groups, last_e, test_size=1 / 3)
    g_tr = set(groups[tr_i])
    g_te = set(groups[te_i])
    assert g_tr.isdisjoint(g_te)
    assert g_te == {3}


def test_temporal_repro_matches_train_logic():
    from pathlib import Path

    import joblib
    import pytest

    from training.reproduce_split import reproducible_train_test_indices, load_xy_groups_from_artifact
    from utils.config import MODEL_PATH

    p = Path(MODEL_PATH)
    if not p.exists():
        pytest.skip("no model.pkl")
    art = joblib.load(p)
    fe = art.get("feature_engineering") or {}
    if fe.get("split_method") != "temporal_patient":
        pytest.skip("artifact not temporal-split")
    X, y, groups = load_xy_groups_from_artifact(art)
    tr_i, te_i = reproducible_train_test_indices(
        X,
        y,
        groups,
        split_method="temporal_patient",
        test_size=float(fe.get("test_size", 0.2)),
        random_state=int(fe.get("random_state", 42)),
        fe=fe,
    )
    assert set(groups[tr_i]).isdisjoint(set(groups[te_i]))
