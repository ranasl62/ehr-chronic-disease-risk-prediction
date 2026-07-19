"""Backend contract for every endpoint the Angular workbench UI calls.

Keeps API shapes stable for Home, Datasets, Train, Results, Analytics,
Predict, and Config pages without requiring a browser.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def _fake_artifact():
    m = DummyClassifier(strategy="prior").fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    return {
        "model": m,
        "feature_columns": ["w7d_glucose", "w7d_age"],
        "model_kind": "dummy",
        "calibrated": False,
        "feature_engineering": {"windows_days": [7, 30, 180]},
        "feature_importance": {"w7d_glucose": 0.6, "w7d_age": 0.4},
        "shap_background": None,
        "input_stats": {
            "w7d_glucose": {"median": 110.0, "p05": 90.0, "p95": 160.0, "mean": 115.0},
            "w7d_age": {"median": 55.0, "p05": 40.0, "p95": 75.0, "mean": 56.0},
        },
        "metrics": {"roc_auc": 0.72, "pr_auc": 0.55, "brier": 0.21},
    }


@pytest.fixture()
def client_with_model():
    from api.main import app, artifact_dep, get_artifact

    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = _fake_artifact
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_health_and_ready(client):
    h = client.get("/health")
    assert h.status_code == 200
    assert "status" in h.json()
    r = client.get("/v1/ready")
    assert r.status_code == 200
    assert "ready" in r.json()


def test_meta_matches_ui_expectations(client):
    r = client.get("/v1/meta")
    assert r.status_code == 200
    js = r.json()
    assert "documentation" in js or "clinical_use" in js or "disclaimer" in js


def test_workspace_status_home_page(client):
    r = client.get("/v1/workspace/status")
    assert r.status_code == 200
    js = r.json()
    for key in ("api_ok", "checklist", "recent_jobs"):
        assert key in js


def test_datasets_list_and_health_profile(client):
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    datasets = r.json()["datasets"]
    assert isinstance(datasets, list)
    assert any(d.get("exists") for d in datasets)

    path = next(d["path"] for d in datasets if d.get("id") == "ehr_data" and d.get("exists"))
    health = client.get("/v1/datasets/health", params={"path": path})
    assert health.status_code == 200
    hj = health.json()
    assert "health" in hj
    assert "n_rows" in hj

    profile = client.get("/v1/datasets/profile", params={"path": path})
    assert profile.status_code == 200
    pj = profile.json()
    assert pj["n_rows"] > 0
    assert "columns" in pj
    assert "label_counts" in pj or "age_band_counts" in pj or "numeric_preview" in pj


def test_dataset_profile_rejects_empty_path(client):
    r = client.get("/v1/datasets/profile", params={"path": ""})
    assert r.status_code in (400, 404, 422)


def test_upload_csv_datasets_page(client):
    csv = b"patient_id,timestamp,glucose,age,label\n1,2023-01-01,100,50,0\n2,2023-01-02,140,60,1\n"
    r = client.post(
        "/v1/datasets/upload",
        files={"file": ("ui_upload_test.csv", csv, "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["path"].endswith("ui_upload_test.csv")
    assert r.json()["n_rows"] == 2


def test_form_import_datasets_page(client):
    r = client.post(
        "/v1/datasets/from-form",
        json={
            "name": "ui_form_contract.csv",
            "rows": [
                {"patient_id": 9, "timestamp": "2023-01-01", "glucose": 101, "age": 44, "label": 0},
                {"patient_id": 10, "timestamp": "2023-01-02", "glucose": 150, "age": 61, "label": 1},
            ],
        },
    )
    assert r.status_code == 200, r.text


def test_tasks_train_page(client):
    r = client.get("/v1/tasks")
    assert r.status_code == 200
    assert "tasks" in r.json()
    assert isinstance(r.json()["tasks"], list)


def test_reports_summary_results_analytics(client):
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    js = r.json()
    assert "files" in js
    assert isinstance(js["files"], list)


def test_results_zip_download(client):
    r = client.get("/v1/reports/download.zip")
    assert r.status_code == 200
    assert len(r.content) > 20
    ctype = r.headers.get("content-type", "")
    assert "zip" in ctype or "octet" in ctype


def test_schema_metrics_predict_page(client_with_model):
    schema = client_with_model.get("/v1/model/schema")
    assert schema.status_code == 200
    sj = schema.json()
    assert sj["feature_columns"] == ["w7d_glucose", "w7d_age"]
    assert sj["model_kind"] == "dummy"

    metrics = client_with_model.get("/v1/model/metrics")
    assert metrics.status_code == 200

    pred = client_with_model.post(
        "/v1/predict",
        json={
            "features": {"w7d_glucose": 110.0, "w7d_age": 55.0},
            "include_explanation": False,
        },
    )
    assert pred.status_code == 200, pred.text
    pj = pred.json()
    assert "risk_probability" in pj
    assert "risk_level" in pj
    assert 0.0 <= float(pj["risk_probability"]) <= 1.0


def test_predict_rejects_empty_features(client_with_model):
    r = client_with_model.post(
        "/v1/predict",
        json={"features": {}, "include_explanation": False},
    )
    assert r.status_code == 422


def test_workspace_config_and_events_config_page(client):
    g = client.get("/v1/workspace/config")
    assert g.status_code == 200
    cfg = g.json()
    assert isinstance(cfg, dict)

    ev = client.get("/v1/events", params={"limit": 10})
    assert ev.status_code == 200
    assert "events" in ev.json()


def test_job_lookup_404(client):
    r = client.get("/v1/jobs/does-not-exist-ui-contract")
    assert r.status_code in (404, 400)


def test_frontend_api_surface_documented_in_root(client):
    """Root discovery should list the routes the UI depends on."""
    r = client.get("/")
    assert r.status_code == 200
    js = r.json()
    blob = str(js).lower()
    for needle in ("predict", "workspace", "datasets", "reports"):
        assert needle in blob
