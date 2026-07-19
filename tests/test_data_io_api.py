"""Multi-import and results ZIP API tests."""

from fastapi.testclient import TestClient


def test_form_import_profile_and_zip():
    from api.main import app

    client = TestClient(app)
    r = client.post(
        "/v1/datasets/from-form",
        json={
            "name": "api_form.csv",
            "rows": [
                {"patient_id": 1, "timestamp": "2023-01-01", "glucose": 100, "label": 0},
                {"patient_id": 1, "timestamp": "2023-06-01", "glucose": 120, "label": 0},
                {"patient_id": 2, "timestamp": "2023-01-01", "glucose": 140, "label": 1},
            ],
        },
    )
    assert r.status_code == 200, r.text
    path = r.json()["path"]
    pr = client.get("/v1/datasets/profile", params={"path": path})
    assert pr.status_code == 200
    assert pr.json()["n_rows"] == 3
    z = client.get("/v1/reports/download.zip")
    assert z.status_code == 200
    assert "zip" in z.headers.get("content-type", "")
    assert len(z.content) > 50
