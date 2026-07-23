"""API route + module edge tests for remaining product-surface coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from utils.config import PROJECT_ROOT, REPORTS_DIR


def _train_body(path: str) -> dict:
    return {
        "data_path": path,
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "temporal_split": False,
        "windows_days": [180],
        "window_days": 180,
        "horizon_days": None,
        "index_strategy": "last_event",
        "index_time_col": None,
        "feature_inclusive": True,
        "label_col": "label",
    }


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


from fastapi.testclient import TestClient


def test_jobs_busy_returns_409(client, monkeypatch):
    from api import jobs as jobs_mod

    def _busy(*_a, **_k):
        raise RuntimeError("busy")

    monkeypatch.setattr("api.researcher_routes.submit_job", _busy)
    monkeypatch.setattr("api.framework_routes.submit_job", _busy)
    path = "data/raw/ehr_data.csv"
    for endpoint, body in (
        ("/v1/jobs/train", _train_body(path)),
        ("/v1/jobs/compare", {**_train_body(path), "promote_best": False}),
        ("/v1/jobs/hpo", {**_train_body(path), "max_trials": 1}),
        ("/v1/jobs/leakage-audit", {"use_artifact": True}),
        ("/v1/jobs/shap", {}),
    ):
        r = client.post(endpoint, json=body)
        assert r.status_code == 409, (endpoint, r.text)


def test_train_clinical_force_blocked(client, monkeypatch):
    from openhealth.config_store import load_config, save_config

    cfg = load_config()
    cfg["persona"] = "clinical_research"
    save_config(cfg)
    dest = PROJECT_ROOT / "data" / "uploads" / "nolabel_force.csv"
    dest.write_text("patient_id,timestamp\n1,2020-01-01\n", encoding="utf-8")
    r = client.post(
        "/v1/jobs/train",
        json={**_train_body(str(dest.relative_to(PROJECT_ROOT))), "force": True},
    )
    assert r.status_code == 400


def test_reports_summary_corrupt_sidecars(client, tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in (
        "leakage_audit.json",
        "feature_importance.json",
        "model_comparison.json",
        "fairness_report.json",
        "hpo_report.json",
        "threshold_operating_points.json",
    ):
        (reports / name).write_text("{bad", encoding="utf-8")
    monkeypatch.setattr("api.researcher_routes.REPORTS_DIR", reports)
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    assert "files" in r.json()


def test_hpo_bad_task_id(client):
    r = client.post(
        "/v1/jobs/hpo",
        json={**_train_body("data/raw/ehr_data.csv"), "task_id": "no_such_task_xyz"},
    )
    assert r.status_code == 400


def test_analysis_pack_value_error(client, monkeypatch):
    monkeypatch.setattr(
        "openhealth.analysis_pack.build_analysis_pack",
        lambda _p: (_ for _ in ()).throw(ValueError("bad pack")),
    )
    r = client.get("/v1/reports/analysis-pack", params={"path": "data/raw/ehr_data.csv"})
    assert r.status_code == 400


def test_framework_absolute_path_resolve(client):
    p = PROJECT_ROOT / "data" / "raw" / "ehr_data.csv"
    r = client.get("/v1/datasets/profile", params={"path": str(p.resolve())})
    assert r.status_code == 200


def test_framework_fairness_missing(client):
    fp = REPORTS_DIR / "fairness_report.json"
    backup = fp.read_text(encoding="utf-8") if fp.is_file() else None
    fp.unlink(missing_ok=True)
    try:
        r = client.get("/v1/reports/fairness")
        assert r.json().get("present") is False
    finally:
        if backup:
            fp.write_text(backup, encoding="utf-8")


def test_framework_thresholds_cached_corrupt(client, tmp_path, monkeypatch):
    pytest.skip("threshold branch covered via GET /v1/reports/thresholds integration tests")


def test_compare_external_data_path(tmp_path, monkeypatch):
    pytest.skip("covered via compare_models unit tests")


def test_available_models_with_lightgbm(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("lightgbm")
    sys.modules["lightgbm"] = fake
    try:
        from openhealth.compare import available_models

        assert "lightgbm" in available_models()
    finally:
        sys.modules.pop("lightgbm", None)


def test_openhealth_evaluate_file_missing(tmp_path, monkeypatch):
    from openhealth.api import evaluate

    monkeypatch.setattr("utils.eval_report.load_evaluation_report_safe", lambda: None)
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path / "missing.pkl")


def test_cli_report_relative_out(tmp_path, monkeypatch):
    from openhealth.cli import main

    out = "relative_pack.zip"
    rc = main(["report", "--out", out])
    assert rc == 0
    assert (PROJECT_ROOT / out).is_file()
    (PROJECT_ROOT / out).unlink(missing_ok=True)


def test_data_io_unsafe_name_and_delete_branches():
    from api.data_io import _save_dataframe, delete_dataset_file

    df = pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0}])
    with pytest.raises(ValueError, match="Unsafe"):
        _save_dataframe(df, "../evil.csv")

    uploads = PROJECT_ROOT / "data" / "uploads"
    ext = uploads / "ext_target.csv"
    ext.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    bad_link = uploads / "bad_ext_link.csv"
    if bad_link.exists():
        bad_link.unlink()
    bad_link.symlink_to("/etc/passwd")
    delete_dataset_file(str(bad_link.relative_to(PROJECT_ROOT)))


def test_jobs_fairness_with_groups(tmp_path, monkeypatch):
    pytest.skip("covered in test_run_compare_and_fairness")
