"""Demo vs user dataset separation (data/demo vs data/uploads)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from utils.config import PROJECT_ROOT, resolve_training_data_path


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def test_demo_files_live_under_data_demo():
    assert (PROJECT_ROOT / "data" / "demo" / "ehr_data.csv").is_file()
    assert (PROJECT_ROOT / "data" / "demo" / "sample_ehr.csv").is_file()


def test_legacy_raw_demo_path_falls_back_to_real_demo_file(tmp_path: Path):
    demo = tmp_path / "data" / "demo"
    demo.mkdir(parents=True)
    expected = demo / "ehr_data.csv"
    expected.write_text("patient_id,timestamp,label\n1,2024-01-01,0\n", encoding="utf-8")

    resolved = resolve_training_data_path(
        "data/raw/ehr_data.csv",
        project_root=tmp_path,
    )

    assert resolved == expected.resolve()


def test_container_recorded_demo_path_resolves_under_current_project(tmp_path: Path):
    demo = tmp_path / "data" / "demo"
    demo.mkdir(parents=True)
    expected = demo / "ehr_data.csv"
    expected.write_text("patient_id,timestamp,label\n1,2024-01-01,0\n", encoding="utf-8")

    resolved = resolve_training_data_path(
        "/stale-container/data/demo/ehr_data.csv",
        project_root=tmp_path,
    )

    assert resolved == expected.resolve()


def test_list_datasets_includes_demo_by_default(client):
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    js = r.json()
    assert js.get("include_demo") is True
    assert js.get("demo_root") == "data/demo"
    assert js.get("uploads_root") == "data/uploads"
    paths = [d["path"] for d in js["datasets"] if d.get("bundled")]
    assert any(p.startswith("data/demo/") for p in paths)
    assert all(d.get("category") == "demo" for d in js["datasets"] if d.get("bundled"))


def test_list_datasets_can_hide_demos(client):
    r = client.get("/v1/datasets", params={"include_demo": False})
    assert r.status_code == 200
    js = r.json()
    assert js.get("include_demo") is False
    assert all(not d.get("bundled") for d in js["datasets"])
    assert all(d.get("category") == "user" or d["path"].startswith("data/uploads/") for d in js["datasets"])


def test_custom_task_points_at_demo_path(client):
    r = client.get("/v1/tasks")
    assert r.status_code == 200
    custom = next(t for t in r.json()["tasks"] if t["id"] == "custom")
    assert custom["suggested_path"] == "data/demo/ehr_data.csv"
    assert custom["target_column"] == "label"
