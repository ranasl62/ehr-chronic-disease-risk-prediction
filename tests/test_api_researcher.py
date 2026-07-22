"""Researcher workbench API tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from utils.config import PROJECT_ROOT


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def test_workspace_status(client):
    r = client.get("/v1/workspace/status")
    assert r.status_code == 200
    js = r.json()
    assert js["api_ok"] is True
    assert "checklist" in js
    assert "demo_dataset" in js["checklist"]


def test_list_datasets(client):
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    ds = r.json()["datasets"]
    ids = {d["id"] for d in ds}
    assert "ehr_data" in ids
    assert any(d["exists"] for d in ds if d["id"] == "ehr_data")


def test_reports_summary(client):
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    assert "files" in r.json()


def test_train_job_tiny_demo(client, wait_jobs_idle):
    """End-to-end train on tiny longitudinal demo (may take a few seconds)."""
    wait_jobs_idle()
    body = {
        "data_path": "data/demo/ehr_data.csv",
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "windows_days": [7, 30, 180],
    }
    r = client.post("/v1/jobs/train", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    status = "queued"
    for _ in range(60):
        jr = client.get(f"/v1/jobs/{job_id}")
        assert jr.status_code == 200
        status = jr.json()["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert status == "succeeded", client.get(f"/v1/jobs/{job_id}").json()
    assert Path(PROJECT_ROOT / "model.pkl").is_file()


def test_train_job_legacy_raw_demo_path_is_accepted(client, wait_jobs_idle):
    """Legacy raw demo references remain usable after the demo files moved."""
    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "split_by_patient": True,
            "windows_days": [7, 30, 180],
        },
    )
    assert r.status_code == 200, r.text
    wait_jobs_idle()


def test_train_job_missing_path_returns_clear_not_found(client):
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/uploads/does-not-exist.csv",
            "data_format": "longitudinal",
        },
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "data not found: data/uploads/does-not-exist.csv"


def test_upload_rejects_non_csv(client):
    r = client.post(
        "/v1/datasets/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
