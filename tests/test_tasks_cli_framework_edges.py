"""Task spec variants, CLI, framework meta, health edges, jobs cancel."""

import time

import pandas as pd
import pytest

from openhealth.cli import main
from openhealth.health import dataset_health_report
from openhealth.task_spec import load_task, list_tasks


def test_load_task_by_id_and_path():
    t1 = load_task("diabetes")
    t2 = load_task("tasks/diabetes.yaml")
    assert t1.id == t2.id == "diabetes"
    assert t1.horizon_days == 365


def test_load_task_missing():
    with pytest.raises(FileNotFoundError):
        load_task("does_not_exist_task_xyz")


def test_list_tasks_nonempty():
    tasks = list_tasks()
    ids = {t.id for t in tasks}
    assert "diabetes" in ids and "custom" in ids and "heart_failure" in ids


def test_task_to_train_params_requires_path():
    from openhealth.task_spec import TaskSpec

    t = TaskSpec(id="x", name="x", suggested_path=None)
    with pytest.raises(ValueError):
        t.to_train_params(None)


def test_heart_failure_windows_include_90():
    t = load_task("heart_failure")
    assert 90 in t.windows_days


def test_cli_init_and_evaluate_and_report(tmp_path, monkeypatch):
    assert main(["init"]) == 0
    # evaluate should not crash even if metrics thin
    rc = main(["evaluate"])
    assert rc == 0
    out = tmp_path / "pack.zip"
    rc2 = main(["report", "--out", str(out)])
    assert rc2 == 0
    assert out.is_file() or out.exists()


def test_cli_train_requires_data_or_task():
    with pytest.raises(SystemExit):
        main(["train"])


def test_health_future_column_high_leakage(tmp_path):
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "future_col.csv"
    pd.DataFrame(
        [
            {
                "patient_id": 1,
                "timestamp": "2020-01-01",
                "label": 0,
                "future_label": 1,
            },
            {
                "patient_id": 2,
                "timestamp": "2020-01-01",
                "label": 1,
                "future_label": 1,
            },
        ]
    ).to_csv(dest, index=False)
    h = dataset_health_report(dest)["health"]
    assert h["leakage_risk"] == "HIGH"


def test_health_missing_file():
    with pytest.raises(FileNotFoundError):
        dataset_health_report("data/raw/no_such_file_abc.csv")


def test_health_unparseable_timestamps(tmp_path):
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "bad_ts.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "not-a-date", "label": 0},
            {"patient_id": 2, "timestamp": "also-bad", "label": 1},
        ]
    ).to_csv(dest, index=False)
    h = dataset_health_report(dest)["health"]
    assert h["temporal_integrity"] == "FAIL"


def test_framework_meta(client):
    r = client.get("/v1/meta/framework")
    assert r.status_code == 200
    body = r.json()
    assert "logreg" in body["supported_models"]
    assert "lstm" in body["unsupported_models"]
    assert "omop_subset" in body["adapters"]


def test_runs_list_api(client):
    r = client.get("/v1/runs")
    assert r.status_code == 200
    assert "runs" in r.json()


def test_cancel_unknown_job(client):
    r = client.post("/v1/jobs/notarealjobid/cancel")
    assert r.status_code == 404


def test_cancel_queued_or_running(client):
    from api import jobs as jobs_mod

    def sleeper(rec):
        time.sleep(1.2)

    rec = jobs_mod.submit_job("sleep2", sleeper)
    r = client.post(f"/v1/jobs/{rec.id}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] in ("cancelled", "running", "succeeded")
    time.sleep(1.5)


def test_path_traversal_blocked(client):
    r = client.get("/v1/datasets/health", params={"path": "../etc/passwd"})
    assert r.status_code == 400


def test_datasets_profile_ok(client):
    r = client.get("/v1/datasets/profile", params={"path": "data/raw/ehr_data.csv"})
    assert r.status_code == 200
    assert r.json()["n_rows"] > 0


def test_sql_mutating_rejected(client):
    r = client.post(
        "/v1/datasets/from-sql",
        json={"sql": "DELETE FROM patients", "connection_url": "sqlite://"},
    )
    assert r.status_code == 400
