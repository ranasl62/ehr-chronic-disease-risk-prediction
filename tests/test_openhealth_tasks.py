"""Task YAML + health + tasks API."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from openhealth.health import dataset_health_report
from openhealth.task_spec import list_tasks, load_task


def test_load_diabetes_task():
    t = load_task("diabetes")
    assert t.id == "diabetes"
    assert t.horizon_days == 365
    assert t.index_time_col == "index_time"
    assert 7 in t.windows_days
    params = t.to_train_params()
    assert "paper_synthetic" in params["data_path"]


def test_list_tasks_includes_custom():
    ids = {t.id for t in list_tasks()}
    assert "diabetes" in ids
    assert "custom" in ids


def test_dataset_health_demo():
    report = dataset_health_report("data/raw/ehr_data.csv")
    h = report["health"]
    assert h["ready_for_training"] is True
    assert h["temporal_integrity"] == "PASS"
    assert h["leakage_risk"] in ("LOW", "MEDIUM", "HIGH")


def test_tasks_and_health_api():
    client = TestClient(app)
    r = client.get("/v1/tasks")
    assert r.status_code == 200
    assert any(t["id"] == "diabetes" for t in r.json()["tasks"])
    r2 = client.get("/v1/datasets/health", params={"path": "data/raw/ehr_data.csv"})
    assert r2.status_code == 200
    assert "health" in r2.json()
    assert Path("tasks/diabetes.yaml").is_file()
