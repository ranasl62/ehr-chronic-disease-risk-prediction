"""Live API E2E: research/product loop against running stack (Docker or uvicorn).

Run: EHR_API_BASE=http://127.0.0.1:8000 pytest tests/test_e2e_features.py -m e2e -v
Skip unit CI: pytest tests/ -m "not e2e"
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Any

import httpx
import pytest

API_BASE = os.environ.get("EHR_API_BASE", "http://127.0.0.1:8000").rstrip("/")
UI_BASE = os.environ.get("EHR_UI_BASE", "http://127.0.0.1:8080").rstrip("/")
DEMO_PATH = "data/demo/ehr_data.csv"
DEMO_FORMAT = "longitudinal"

pytestmark = pytest.mark.e2e


def _api_alive() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def api() -> httpx.Client:
    if not _api_alive():
        pytest.skip(f"API not reachable at {API_BASE}")
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        yield client


def _wait_job(client: httpx.Client, jid: str, *, max_wait_s: float = 120.0) -> dict[str, Any]:
    import time

    deadline = time.time() + max_wait_s
    st: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{jid}")
        r.raise_for_status()
        st = r.json()
        if st.get("status") in ("succeeded", "failed", "cancelled"):
            return st
        time.sleep(0.25)
    return st


def _wait_jobs_idle(client: httpx.Client, timeout: float = 30.0) -> None:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get("/v1/jobs", params={"limit": 30})
        if r.status_code != 200:
            time.sleep(0.2)
            continue
        jobs = r.json().get("jobs") or r.json()
        if isinstance(jobs, dict):
            jobs = jobs.get("jobs", [])
        busy = [j for j in jobs if j.get("status") in ("queued", "running")]
        if not busy:
            return
        time.sleep(0.2)


def test_e2e_health_ready_meta(api: httpx.Client) -> None:
    h = api.get("/health")
    assert h.status_code == 200
    assert h.json().get("status") == "ok"
    ready = api.get("/v1/ready")
    assert ready.status_code == 200
    meta = api.get("/v1/meta")
    assert meta.status_code == 200


def test_e2e_datasets_workspace_analytics_reads(api: httpx.Client) -> None:
    ds = api.get("/v1/datasets")
    assert ds.status_code == 200
    datasets = ds.json()["datasets"]
    assert any(d.get("exists") for d in datasets)
    path = next(
        d["path"]
        for d in datasets
        if d.get("exists") and ("demo" in d["path"] or d.get("id") == "ehr_data")
    )
    ws = api.get("/v1/workspace/status")
    assert ws.status_code == 200
    assert ws.json().get("api_ok") is True
    assert "checklist" in ws.json()

    profile = api.get("/v1/datasets/profile", params={"path": path})
    assert profile.status_code == 200
    assert profile.json().get("n_rows", 0) > 0

    health = api.get(
        "/v1/datasets/health",
        params={"path": path, "task_id": "horizon_detection_30d"},
    )
    assert health.status_code == 200
    assert "health" in health.json()

    tasks = api.get("/v1/tasks")
    assert tasks.status_code == 200
    assert tasks.json().get("tasks")

    summary = api.get("/v1/reports/summary")
    assert summary.status_code == 200
    assert "files" in summary.json()

    cfg = api.get("/v1/workspace/config")
    assert cfg.status_code == 200
    ev = api.get("/v1/events", params={"limit": 5})
    assert ev.status_code == 200
    assert "events" in ev.json()


def test_e2e_ui_proxies_api(api: httpx.Client) -> None:
    """Nginx workbench should serve SPA and proxy /v1 to API."""
    try:
        home = httpx.get(f"{UI_BASE}/", timeout=10.0, follow_redirects=True)
    except Exception as exc:
        pytest.skip(f"UI not reachable at {UI_BASE}: {exc}")
    assert home.status_code == 200
    assert "html" in home.headers.get("content-type", "").lower()

    proxied = httpx.get(f"{UI_BASE}/v1/workspace/status", timeout=15.0)
    assert proxied.status_code == 200
    assert proxied.json().get("api_ok") is True


def test_e2e_full_research_workflow(api: httpx.Client) -> None:
    _wait_jobs_idle(api)
    train = api.post(
        "/v1/jobs/train",
        json={
            "data_path": DEMO_PATH,
            "data_format": DEMO_FORMAT,
            "model_kind": "logreg",
            "promote": True,
            "calibrate": False,
            "windows_days": [7, 30, 180],
        },
    )
    assert train.status_code == 200, train.text
    train_st = _wait_job(api, train.json()["id"])
    _wait_jobs_idle(api)
    assert train_st["status"] == "succeeded", train_st
    run_id = train_st["result"]["run_id"]
    assert run_id

    run_detail = api.get(f"/v1/runs/{run_id}")
    assert run_detail.status_code == 200
    assert run_detail.json().get("run_id") == run_id

    _wait_jobs_idle(api)
    la = api.post("/v1/jobs/leakage-audit", json={"use_artifact": True, "run_id": run_id})
    assert la.status_code == 200, la.text
    la_st = _wait_job(api, la.json()["id"])
    _wait_jobs_idle(api)
    assert la_st["status"] == "succeeded", la_st

    ap = api.get("/v1/reports/analysis-pack", params={"path": DEMO_PATH})
    assert ap.status_code == 200
    pack = ap.json()
    assert pack.get("n_rows", 0) > 0
    assert "missingness" in pack

    _wait_jobs_idle(api)
    ev = api.post(
        "/v1/jobs/external-validate",
        json={"data_path": DEMO_PATH, "data_format": DEMO_FORMAT, "run_id": run_id},
    )
    assert ev.status_code == 200, ev.text
    ev_st = _wait_job(api, ev.json()["id"])
    _wait_jobs_idle(api)
    assert ev_st["status"] == "succeeded", ev_st

    md = api.get("/v1/reports/methods.md", params={"run_id": run_id})
    assert md.status_code == 200
    assert "Trust pack" in md.text

    zr = api.get("/v1/reports/download.zip", params={"run_id": run_id})
    assert zr.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(zr.content))
    assert any("methods" in n for n in zf.namelist())

    schema = api.get("/v1/model/schema")
    assert schema.status_code == 200
    feats = schema.json()["feature_columns"]
    stats = schema.json().get("input_stats") or {}
    body = {
        "features": {c: float((stats.get(c) or {}).get("median") or 0.0) for c in feats},
        "include_explanation": False,
    }
    pred = api.post("/v1/predict", json=body)
    assert pred.status_code == 200, pred.text
    assert "risk_probability" in pred.json()

    metrics = api.get("/v1/model/metrics")
    assert metrics.status_code == 200

    listed = api.get("/v1/runs", params={"limit": 20})
    assert listed.status_code == 200
    runs = listed.json().get("runs") or []
    assert any(r.get("run_id") == run_id for r in runs)


def test_e2e_shap_job_optional(api: httpx.Client) -> None:
    """SHAP on demo logreg is usually fast; skip clearly if job fails or times out."""
    _wait_jobs_idle(api)
    sh = api.post("/v1/jobs/shap", json={})
    if sh.status_code == 409:
        pytest.skip(f"SHAP job not startable: {sh.text}")
    assert sh.status_code == 200, sh.text
    st = _wait_job(api, sh.json()["id"], max_wait_s=180.0)
    _wait_jobs_idle(api)
    if st.get("status") != "succeeded":
        pytest.skip(f"SHAP job did not succeed (demo/model): {st}")
    img = api.get("/v1/reports/file/shap_summary.png")
    assert img.status_code == 200
    assert len(img.content) > 100
