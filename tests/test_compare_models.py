"""Lightweight compare API smoke (single model)."""

from fastapi.testclient import TestClient

from api.main import app
from openhealth.compare import available_models, compare_models


def test_available_models_includes_logreg():
    assert "logreg" in available_models()


def test_compare_single_model_promotes(tmp_path):
    summary = compare_models(
        data_path="data/raw/ehr_data.csv",
        data_format="longitudinal",
        models=["logreg"],
        calibrate=False,
        split_by_patient=True,
        windows_days=(7, 30, 180),
        out_dir=tmp_path / "compare",
        promote_best=True,
    )
    assert summary["selected_model"] == "logreg"
    assert summary["comparison"][0]["selected"] is True


def test_compare_job_endpoint_queues():
    client = TestClient(app)
    # May 409 if another job running — accept 200 or 409
    r = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "models": ["logreg"],
            "calibrate": False,
            "split_by_patient": True,
            "windows_days": [180],
            "promote_best": False,
        },
    )
    assert r.status_code in (200, 409)
