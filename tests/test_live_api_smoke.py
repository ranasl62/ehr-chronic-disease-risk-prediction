"""Live HTTP smoke against a running API (Docker/local).

Skipped automatically when the API is unreachable so unit CI stays green.
"""

from __future__ import annotations

import os

import pytest
import urllib.error
import urllib.request


API = os.environ.get("EHR_API_BASE", "http://localhost:8000").rstrip("/")


def _alive() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _alive(), reason=f"API not reachable at {API}")


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        import json

        return json.loads(r.read().decode())


def test_live_workspace_and_datasets():
    st = _get("/v1/workspace/status")
    assert st["api_ok"] is True
    ds = _get("/v1/datasets")
    assert any(d.get("exists") for d in ds["datasets"])


def test_live_profile_and_reports():
    path = "data/raw/ehr_data.csv"
    pr = _get(f"/v1/datasets/profile?path={path}")
    assert pr["n_rows"] > 0
    rep = _get("/v1/reports/summary")
    assert "files" in rep


def test_live_schema_and_predict():
    import json
    import urllib.request

    schema = _get("/v1/model/schema")
    feats = schema["feature_columns"]
    assert feats
    stats = schema.get("input_stats") or {}
    body = {
        "features": {c: float((stats.get(c) or {}).get("median") or 0.0) for c in feats},
        "include_explanation": False,
    }
    req = urllib.request.Request(
        f"{API}/v1/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        out = json.loads(r.read().decode())
    assert "risk_probability" in out
    assert "risk_level" in out
