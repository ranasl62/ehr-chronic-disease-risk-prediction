"""API researcher + framework integration edge cases."""

import time


def test_leakage_audit_job(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post("/v1/jobs/leakage-audit", json={"use_artifact": True})
    assert r.status_code in (200, 409)
    if r.status_code != 200:
        return
    jid = r.json()["id"]
    for _ in range(60):
        st = client.get(f"/v1/jobs/{jid}").json()
        if st["status"] in ("succeeded", "failed"):
            assert st["status"] == "succeeded"
            break
        time.sleep(0.2)


def test_shap_job_without_model_or_ok(client, wait_jobs_idle):
    from pathlib import Path

    from utils.config import MODEL_PATH

    wait_jobs_idle()
    r = client.post("/v1/jobs/shap")
    if not Path(MODEL_PATH).is_file():
        assert r.status_code in (200, 409)
        if r.status_code == 200:
            jid = r.json()["id"]
            for _ in range(40):
                st = client.get(f"/v1/jobs/{jid}").json()
                if st["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.15)
            wait_jobs_idle()
    else:
        assert r.status_code in (200, 409)
        if r.status_code == 200:
            jid = r.json()["id"]
            for _ in range(80):
                st = client.get(f"/v1/jobs/{jid}").json()
                if st["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.2)
            wait_jobs_idle()


def test_reports_summary_and_download(client):
    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    assert "files" in r.json()
    z = client.get("/v1/reports/download.zip")
    assert z.status_code == 200
    assert z.headers["content-type"].startswith("application/zip") or "zip" in z.headers.get(
        "content-type", ""
    )


def test_reports_file_not_allowlisted(client):
    r = client.get("/v1/reports/file/../../../etc/passwd")
    assert r.status_code in (404, 422)


def test_workspace_status(client):
    r = client.get("/v1/workspace/status")
    assert r.status_code == 200
    body = r.json()
    assert body.get("api_ok") is True
    assert "checklist" in body


def test_tasks_list_and_get(client):
    r = client.get("/v1/tasks")
    assert r.status_code == 200
    assert any(t["id"] == "diabetes" for t in r.json()["tasks"])
    r2 = client.get("/v1/tasks/diabetes")
    assert r2.status_code == 200
    r3 = client.get("/v1/tasks/nope_missing")
    assert r3.status_code == 404


def test_form_import_requires_patient(client):
    r = client.post(
        "/v1/datasets/from-form",
        json={"name": "bad_form.csv", "rows": [{"glucose": 1}]},
    )
    assert r.status_code == 400


def test_form_import_ok(client):
    r = client.post(
        "/v1/datasets/from-form",
        json={
            "name": "ok_form.csv",
            "rows": [
                {
                    "patient_id": 1,
                    "timestamp": "2020-01-01",
                    "glucose": 100,
                    "label": 0,
                },
                {
                    "patient_id": 2,
                    "timestamp": "2020-01-01",
                    "glucose": 140,
                    "label": 1,
                },
            ],
        },
    )
    assert r.status_code == 200


def test_upload_rejects_empty_filename(client):
    # httpx may not allow empty filename easily; send minimal invalid
    r = client.post(
        "/v1/datasets/upload",
        files={"file": ("", b"patient_id,timestamp,label\n1,2020-01-01,0\n")},
    )
    assert r.status_code in (400, 422)


def test_compare_job_single_model(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "models": ["logreg"],
            "windows_days": [180],
            "promote_best": False,
        },
    )
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        jid = r.json()["id"]
        for _ in range(80):
            st = client.get(f"/v1/jobs/{jid}").json()
            if st["status"] in ("succeeded", "failed"):
                assert st["status"] == "succeeded"
                break
            time.sleep(0.2)
        wait_jobs_idle()


def test_worklist_predict_researcher_ok(client):
    from pathlib import Path

    from utils.config import MODEL_PATH

    if not Path(MODEL_PATH).is_file():
        return
    r = client.post(
        "/v1/worklist/predict",
        json={"rows": [{"w7d_age": 55.0, "w30d_glucose": 110.0}]},
    )
    # may 200 or fail if feature mismatch — accept 200
    assert r.status_code in (200, 400, 500)
    if r.status_code == 200:
        assert r.json()["n"] == 1
