"""Gap-closure API: runs detail, fairness report, HPO job, thresholds, tasks columns."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from openhealth.runs import ensure_run, get_run, new_run_id, write_run_meta
from openhealth.task_spec import load_task
from training.evaluate import threshold_operating_points
from utils.config import PROJECT_ROOT, REPORTS_DIR


def test_task_required_columns_readmission():
    spec = load_task("readmission_30d")
    pub = spec.to_public()
    assert "index_time" in pub["required_columns"]
    assert "patient_id" in pub["required_columns"]
    assert pub["description"]


def test_get_run_detail_and_list_api(client):
    rid = new_run_id("gap")
    ensure_run(rid)
    write_run_meta(rid, {"kind": "train", "model_kind": "logreg"})
    (Path(ensure_run(rid)) / "evaluation_report.json").write_text(
        json.dumps({"metrics": {"roc_auc": 0.77}, "meta": {"model_kind": "logreg"}}),
        encoding="utf-8",
    )
    detail = get_run(rid)
    assert detail["run_id"] == rid
    assert detail["metrics"]["roc_auc"] == 0.77

    r = client.get(f"/v1/runs/{rid}")
    assert r.status_code == 200
    assert r.json()["run_id"] == rid

    missing = client.get("/v1/runs/does_not_exist_gap_zzz")
    assert missing.status_code == 404


def test_fairness_report_endpoint_empty(client):
    r = client.get("/v1/reports/fairness")
    assert r.status_code == 200
    body = r.json()
    assert "present" in body


def test_reports_summary_includes_fairness_keys(client):
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    js = r.json()
    assert "fairness" in js
    assert "hpo" in js
    assert "thresholds" in js
    assert "files" in js


def test_methods_md_endpoint(client):
    r = client.get("/v1/reports/methods.md")
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    body = r.text
    assert "Methods note" in body
    assert "research" in body.lower()
    assert "patient care" in body.lower()


def test_threshold_operating_points_helper():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.55, 0.8, 0.9, 0.2])
    rows = threshold_operating_points(y, p, thresholds=[0.5, 0.7])
    assert len(rows) == 2
    assert rows[0]["threshold"] == 0.5
    assert "precision" in rows[0] and "recall" in rows[0]


def test_hpo_job_queues(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/hpo",
        json={
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
        },
    )
    assert r.status_code in (200, 409), r.text
    if r.status_code == 200:
        jid = r.json()["id"]
        assert r.json()["kind"] == "hpo"
        # poll briefly
        for _ in range(60):
            st = client.get(f"/v1/jobs/{jid}").json()
            if st["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.4)
        assert st["status"] in ("succeeded", "failed")
        if st["status"] == "succeeded":
            assert (REPORTS_DIR / "hpo_report.json").is_file()


def test_jobs_list_endpoint(client):
    r = client.get("/v1/jobs")
    assert r.status_code == 200
    assert "jobs" in r.json()


def test_light_hpo_module_runs():
    from training.hpo import run_light_hpo

    data = PROJECT_ROOT / "data" / "raw" / "ehr_data.csv"
    if not data.is_file():
        return
    out = run_light_hpo(
        data_path=data,
        model_kind="logreg",
        data_format="longitudinal",
        windows_days=(7, 30, 180),
        max_trials=2,
        promote_best=False,
    )
    assert out["n_trials"] >= 1
    assert out["best"] is not None
    assert Path(PROJECT_ROOT / out["report_path"]).is_file()
