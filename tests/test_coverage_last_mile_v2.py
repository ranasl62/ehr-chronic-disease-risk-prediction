"""Cover remaining api edge lines for 100% gate."""

from __future__ import annotations

from pathlib import Path

import api.main as main_mod


def test_get_artifact_missing_model(client, monkeypatch, tmp_path):
    missing = tmp_path / "no_model.pkl"
    monkeypatch.setattr(main_mod, "MODEL_PATH", missing)
    main_mod.get_artifact.cache_clear()
    try:
        r = client.get("/v1/model/schema")
        assert r.status_code == 503
    finally:
        main_mod.get_artifact.cache_clear()


def test_worklist_predict_row_value_error(client, monkeypatch):
    def boom(_row):
        raise ValueError("bad row features")

    monkeypatch.setattr("openhealth.api.predict", boom)
    r = client.post("/v1/worklist/predict", json={"rows": [{"w7d_glucose": 1.0}]})
    assert r.status_code == 400
    detail = r.json().get("detail") or {}
    assert "bad row" in str(detail).lower() or detail.get("row_index") == 0
