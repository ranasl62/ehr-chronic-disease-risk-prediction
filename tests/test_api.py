from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from utils.config import MODEL_PATH


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def _fake_artifact():
    m = DummyClassifier(strategy="prior").fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
    return {
        "model": m,
        "feature_columns": ["a"],
        "model_kind": "dummy",
        "calibrated": False,
        "feature_engineering": {},
        "feature_importance": {"a": 1.0},
        "shap_background": None,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    js = r.json()
    assert "status" in js
    assert "model_path" in js
    assert r.headers.get("X-Clinical-Disclaimer")


def test_predict_rejects_nan_at_inference_layer():
    """JSON cannot encode NaN; bad floats can still arrive from Python callers."""
    from inference.predict import predict_row

    with pytest.raises(ValueError, match="finite"):
        predict_row({"a": float("nan")}, artifact=_fake_artifact())


def test_meta_public(client):
    r = client.get("/v1/meta")
    assert r.status_code == 200
    js = r.json()
    assert js["clinical_use"]
    assert "documentation" in js


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


def test_api_key_enforced_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret-key")
    from api.main import app, artifact_dep, get_artifact

    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = _fake_artifact
    try:
        c2 = TestClient(app)
        r = c2.post(
            "/v1/predict",
            json={"features": {"a": 1.0}, "include_explanation": False},
        )
        assert r.status_code == 401
        r2 = c2.post(
            "/v1/predict",
            json={"features": {"a": 1.0}, "include_explanation": False},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert r2.status_code == 200
        assert "risk_probability" in r2.json()
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_schema_with_injected_artifact():
    from api.main import app, artifact_dep, get_artifact

    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = lambda: {
        **_fake_artifact(),
        "feature_columns": ["age", "glucose"],
    }
    try:
        c = TestClient(app)
        r = c.get("/v1/model/schema")
        assert r.status_code == 200
        assert r.json()["feature_columns"] == ["age", "glucose"]
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_model_metrics_503_when_eval_missing(monkeypatch):
    from api.main import app, artifact_dep, get_artifact

    monkeypatch.setattr("api.main.load_evaluation_report_safe", lambda: None)
    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = _fake_artifact
    try:
        c = TestClient(app)
        r = c.get("/v1/model/metrics")
        assert r.status_code == 503
        assert "evaluation_report" in (r.json().get("detail") or "").lower()
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_model_metrics_200_aligned_when_sha_matches(monkeypatch):
    from api.main import app, artifact_dep, get_artifact

    monkeypatch.setattr(
        "api.main.load_evaluation_report_safe",
        lambda: {
            "generated_at_utc": "2020-01-01T00:00:00Z",
            "threshold": 0.5,
            "metrics": {"roc_auc": 0.8},
            "meta": {
                "training_manifest": {"data_sha256": "abc"},
                "feature_engineering": {"split_method": "patient_group"},
            },
        },
    )
    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = lambda: {
        **_fake_artifact(),
        "training_manifest": {"data_sha256": "abc"},
    }
    try:
        c = TestClient(app)
        r = c.get("/v1/model/metrics")
        assert r.status_code == 200
        js = r.json()
        assert js["evaluation_aligned_with_loaded_artifact"] is True
        assert js["metrics"]["roc_auc"] == 0.8
        assert js["meta_summary"]["split_method"] == "patient_group"
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_schema_with_real_artifact_if_present(client):
    from inference.predict import load_artifact

    if not Path(MODEL_PATH).exists():
        pytest.skip("model.pkl not present")
    try:
        load_artifact(MODEL_PATH)
    except Exception as exc:
        pytest.skip(f"model.pkl incompatible with runtime (retrain): {exc}")
    from api.main import get_artifact

    get_artifact.cache_clear()
    r = client.get("/v1/model/schema")
    assert r.status_code == 200
    js = r.json()
    assert "feature_columns" in js
