"""Delete demo or uploaded datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from utils.config import PROJECT_ROOT


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def test_delete_upload(client):
    uploads = PROJECT_ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / "delete_me_unit.csv"
    path.write_text("patient_id,timestamp,label\n1,2023-01-01,0\n", encoding="utf-8")
    rel = str(path.relative_to(PROJECT_ROOT))
    r = client.delete("/v1/datasets", params={"path": rel})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert not path.exists()


def test_delete_missing_idempotent(client):
    r = client.delete("/v1/datasets", params={"path": "data/uploads/no_such_file_zzz.csv"})
    assert r.status_code == 200, r.text
    assert r.json() == {
        "deleted": True,
        "already_absent": True,
        "path": "data/uploads/no_such_file_zzz.csv",
        "requested": "data/uploads/no_such_file_zzz.csv",
    }


def test_delete_outside_data_rejected(client):
    r = client.delete("/v1/datasets", params={"path": "README.md"})
    assert r.status_code == 400


def test_delete_demo_temp_file(client):
    """Delete a disposable file under data/demo (not the shared teaching CSV)."""
    demo = PROJECT_ROOT / "data" / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    path = demo / "tmp_delete_unit.csv"
    path.write_text("patient_id,timestamp,label\n1,2023-01-01,0\n", encoding="utf-8")
    rel = str(path.relative_to(PROJECT_ROOT))
    r = client.delete("/v1/datasets", params={"path": rel})
    assert r.status_code == 200, r.text
    assert not path.exists()


def test_delete_url_encoded_path(client):
    uploads = PROJECT_ROOT / "data" / "uploads"
    path = uploads / "encoded_delete_unit.csv"
    path.write_text("patient_id,timestamp,label\n1,2023-01-01,0\n", encoding="utf-8")

    r = client.delete("/v1/datasets?path=data%2Fuploads%2Fencoded_delete_unit.csv")

    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert not path.exists()


def test_delete_same_upload_twice_is_idempotent(client):
    uploads = PROJECT_ROOT / "data" / "uploads"
    path = uploads / "double_delete_unit.csv"
    path.write_text("patient_id,timestamp,label\n1,2023-01-01,0\n", encoding="utf-8")
    rel = str(path.relative_to(PROJECT_ROOT))

    first = client.delete("/v1/datasets", params={"path": rel})
    second = client.delete("/v1/datasets", params={"path": rel})

    assert first.status_code == 200, first.text
    assert first.json()["already_absent"] is False
    assert second.status_code == 200, second.text
    assert second.json()["already_absent"] is True


def test_delete_protected_demo_readme_rejected(client):
    r = client.delete("/v1/datasets", params={"path": "data/demo/README.md"})

    assert r.status_code == 400
