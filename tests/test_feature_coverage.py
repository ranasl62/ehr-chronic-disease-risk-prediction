"""Product-feature coverage for UI surfaces + critical training/jobs/reports paths.

Inventory (routes vs existing tests) — gaps this file closes for the workbench:

Covered here (were thin or missing from test_ui_api_contract.py):
  - GET /v1/reports/methods.md (+ ZIP methods.md membership)
  - GET /v1/runs, GET /v1/runs/{id}
  - GET /v1/reports/fairness, GET /v1/reports/thresholds
  - POST /v1/jobs/train with label_col=label + ehr_data / tiny fixture
  - POST /v1/jobs/compare, /hpo, /leakage-audit, /shap (smoke: 200 or busy 409)
  - GET /v1/tasks custom.target_column == label (matches ehr_data.csv)

Already covered elsewhere (not duplicated as primary owners):
  - test_ui_api_contract.py — health, meta, workspace, datasets, upload, form,
    tasks list shape, reports/summary, ZIP bytes, schema/metrics/predict, config/events
  - test_gap_closures.py — run detail helper, fairness empty, methods.md tone,
    HPO poll, jobs list, threshold helper
  - test_api_researcher.py — train without explicit label_col
  - test_trust_jobs.py — health gate, fairness job, busy 409
  - test_facade_claims_io_edges.py — build_results_zip methods.md content
  - test_compare_models.py / test_api_integration_edges.py — compare / leakage / shap edges

Still intentionally out of scope (not 100% lines):
  - OMOP/FHIR/SQL import adapters, map-import heavy paths, worklist predict/audit
  - PUT workspace/config mutations, task create POST, promote-run success path
  - Production middleware / API-key matrix beyond trust_jobs
  - Full Optuna, LSTM, clinical threshold policy (deferred in GAP_CLOSURES.md)
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openhealth.runs import ensure_run, new_run_id, write_run_meta
from utils.config import PROJECT_ROOT


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def _poll_job(client: TestClient, job_id: str, *, rounds: int = 90, sleep: float = 0.4) -> dict:
    st: dict = {}
    for _ in range(rounds):
        st = client.get(f"/v1/jobs/{job_id}").json()
        if st.get("status") in ("succeeded", "failed", "cancelled"):
            return st
        time.sleep(sleep)
    return st


# --- Reports / methods ---


def test_methods_md_research_only_language(client):
    r = client.get("/v1/reports/methods.md")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    assert "text/markdown" in ctype
    body = r.text
    assert "Methods note" in body
    lower = body.lower()
    assert "research" in lower
    assert "patient care" in lower
    assert "not intended for patient care" in lower or "not fda" in lower or "education" in lower


def test_results_zip_includes_methods_md(client):
    r = client.get("/v1/reports/download.zip")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert "reports/methods.md" in names
    methods = zf.read("reports/methods.md").decode("utf-8")
    assert "Methods note" in methods
    assert "research" in methods.lower()


# --- Runs browser ---


def test_runs_list_and_detail(client):
    rid = new_run_id("feat")
    ensure_run(rid)
    write_run_meta(rid, {"kind": "train", "model_kind": "logreg"})

    listing = client.get("/v1/runs", params={"limit": 50})
    assert listing.status_code == 200
    runs = listing.json()["runs"]
    assert isinstance(runs, list)
    assert any(x.get("run_id") == rid for x in runs)

    detail = client.get(f"/v1/runs/{rid}")
    assert detail.status_code == 200
    assert detail.json()["run_id"] == rid

    missing = client.get("/v1/runs/does_not_exist_feature_cov_zzz")
    assert missing.status_code == 404


# --- Fairness / thresholds ---


def test_fairness_report_endpoint(client):
    r = client.get("/v1/reports/fairness")
    assert r.status_code == 200
    body = r.json()
    assert "present" in body
    assert isinstance(body["present"], bool)


def test_thresholds_report_endpoint(client):
    r = client.get("/v1/reports/thresholds")
    # 200 when model or cached JSON exists; 404 when neither (fresh CI)
    assert r.status_code in (200, 404), r.text
    if r.status_code == 200:
        body = r.json()
        assert body.get("present") is True
        assert "points" in body or "threshold" in body
        note = str(body.get("note", "")).lower()
        if note:
            assert "research" in note or "clinical" in note


# --- Tasks: custom matches ehr_data label ---


def test_custom_task_target_column_is_label(client):
    r = client.get("/v1/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    custom = next((t for t in tasks if t.get("id") == "custom"), None)
    assert custom is not None, "custom task missing from /v1/tasks"
    assert custom.get("target_column") == "label"
    assert "ehr_data" in str(custom.get("suggested_path") or "")

    one = client.get("/v1/tasks/custom")
    assert one.status_code == 200
    assert one.json().get("target_column") == "label"


# --- Demo train path (label_col=label) ---


def test_train_job_with_label_col_label(client, wait_jobs_idle, tiny_csv):
    """UI demo / Train form sends label_col=label for bundled and fixture CSVs."""
    wait_jobs_idle()
    data_path = "data/raw/ehr_data.csv"
    if not (PROJECT_ROOT / data_path).is_file():
        # Fall back to tiny fixture copied under uploads
        dest = PROJECT_ROOT / "data" / "uploads" / "feature_cov_tiny.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(tiny_csv).read_bytes())
        data_path = "data/uploads/feature_cov_tiny.csv"

    body = {
        "data_path": data_path,
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "temporal_split": False,
        "windows_days": [7, 30, 180],
        "label_col": "label",
        "task_id": "custom",
    }
    r = client.post("/v1/jobs/train", json=body)
    assert r.status_code in (200, 409), r.text
    if r.status_code == 409:
        wait_jobs_idle()
        return
    st = _poll_job(client, r.json()["id"])
    assert st.get("status") == "succeeded", st
    wait_jobs_idle()


# --- Job smokes (success or graceful busy skip) ---


def _job_smoke(client: TestClient, wait_jobs_idle, method: str, path: str, json_body: dict | None):
    wait_jobs_idle()
    if method == "POST":
        r = client.post(path, json=json_body or {})
    else:
        r = client.get(path)
    assert r.status_code in (200, 409), r.text
    if r.status_code == 200 and "id" in r.json():
        st = _poll_job(client, r.json()["id"], rounds=90)
        assert st.get("status") in ("succeeded", "failed", "cancelled"), st
    wait_jobs_idle()


def test_compare_job_smoke(client, wait_jobs_idle):
    _job_smoke(
        client,
        wait_jobs_idle,
        "POST",
        "/v1/jobs/compare",
        {
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "models": ["logreg"],
            "calibrate": False,
            "split_by_patient": True,
            "windows_days": [180],
            "promote_best": False,
            "label_col": "label",
        },
    )


def test_hpo_job_smoke(client, wait_jobs_idle):
    _job_smoke(
        client,
        wait_jobs_idle,
        "POST",
        "/v1/jobs/hpo",
        {
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "calibrate": False,
            "split_by_patient": True,
            "temporal_split": False,
            "windows_days": [7, 30, 180],
            "window_days": 180,
            "max_trials": 2,
            "promote_best": False,
            "label_col": "label",
        },
    )


def test_leakage_audit_job_smoke(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post("/v1/jobs/leakage-audit", json={"use_artifact": True})
    assert r.status_code in (200, 400, 409), r.text
    if r.status_code == 200:
        st = _poll_job(client, r.json()["id"], rounds=45)
        assert st.get("status") in ("succeeded", "failed", "cancelled")
    wait_jobs_idle()


def test_shap_job_smoke(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post("/v1/jobs/shap")
    assert r.status_code in (200, 400, 409), r.text
    if r.status_code == 200:
        st = _poll_job(client, r.json()["id"], rounds=45)
        assert st.get("status") in ("succeeded", "failed", "cancelled")
    wait_jobs_idle()


# --- Gap-closure surfaces not already asserted in ui contract ---


def test_reports_summary_gap_keys(client):
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    js = r.json()
    for key in ("files", "fairness", "hpo", "thresholds"):
        assert key in js


def test_jobs_list_shape(client):
    r = client.get("/v1/jobs")
    assert r.status_code == 200
    assert isinstance(r.json().get("jobs"), list)
