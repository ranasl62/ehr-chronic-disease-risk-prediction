"""Training split reproduction matches patient-level isolation when configured."""

from pathlib import Path

import joblib
import pytest

from training.reproduce_split import (
    load_xy_groups_from_artifact,
    reproducible_train_test_indices,
    split_train_test_from_artifact,
)
from utils.config import MODEL_PATH


def test_artifact_container_demo_path_resolves_on_host():
    artifact = {
        "feature_engineering": {
            "format": "longitudinal",
            "data_path": "/stale-container/data/demo/ehr_data.csv",
            "window_days": 180,
        }
    }

    X, y, groups = load_xy_groups_from_artifact(artifact)

    assert len(X) == len(y) == len(groups)
    assert len(X) > 0


def test_artifact_missing_data_path_raises_clear_error(tmp_path: Path):
    artifact = {
        "feature_engineering": {
            "format": "tabular",
            "data_path": str(tmp_path / "missing.csv"),
        }
    }

    with pytest.raises(FileNotFoundError, match="Artifact data_path not found"):
        load_xy_groups_from_artifact(artifact)


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="model.pkl not present")
def test_artifact_split_roundtrip_rows():
    art = joblib.load(MODEL_PATH)
    X_tr, X_te, y_tr, y_te, _, _ = split_train_test_from_artifact(art)
    assert len(X_tr) + len(X_te) == len(load_xy_groups_from_artifact(art)[0])
    assert len(y_tr) == len(X_tr) and len(y_te) == len(X_te)


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="model.pkl not present")
def test_patient_group_split_disjoint_when_configured():
    art = joblib.load(MODEL_PATH)
    fe = art.get("feature_engineering") or {}
    if fe.get("split_method") != "patient_group":
        pytest.skip("artifact not trained with --split-by-patient")
    X, y, groups = load_xy_groups_from_artifact(art)
    tr_i, te_i = reproducible_train_test_indices(
        X,
        y,
        groups,
        split_method="patient_group",
        test_size=float(fe.get("test_size", 0.2)),
        random_state=int(fe.get("random_state", 42)),
    )
    g_tr = set(groups[tr_i])
    g_te = set(groups[te_i])
    assert g_tr.isdisjoint(g_te)
