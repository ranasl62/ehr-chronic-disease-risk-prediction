"""Trust: health-gated train, API key, fairness, jobs busy."""

import os
import time

from openhealth.config_store import load_config, save_config


def test_train_health_gate_blocks(client):
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "bad_train.csv"
    dest.write_text("patient_id,timestamp\n1,2023-01-01\n2,2023-01-02\n", encoding="utf-8")
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/uploads/bad_train.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "force": False,
        },
    )
    assert r.status_code == 400


def test_train_clinical_cannot_force(client):
    from utils.config import PROJECT_ROOT

    cfg = load_config()
    cfg["persona"] = "clinical_research"
    cfg["disclaimer_ack"] = True
    save_config(cfg)
    dest = PROJECT_ROOT / "data" / "uploads" / "bad_train2.csv"
    dest.write_text("patient_id,timestamp\n1,2023-01-01\n2,2023-01-02\n", encoding="utf-8")
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/uploads/bad_train2.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "force": True,
        },
    )
    assert r.status_code == 400
    # reset persona
    cfg["persona"] = "researcher"
    save_config(cfg)


def test_api_key_mode(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-test-key")
    # reload security check uses os.environ at request time
    r = client.get("/v1/datasets")
    assert r.status_code == 401
    r2 = client.get("/v1/datasets", headers={"X-API-Key": "secret-test-key"})
    assert r2.status_code == 200
    monkeypatch.delenv("API_KEY", raising=False)


def test_fairness_job_queues_or_busy(client, wait_jobs_idle):
    wait_jobs_idle()
    r = client.post("/v1/jobs/fairness", json={})
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        jid = r.json()["id"]
        for _ in range(30):
            st = client.get(f"/v1/jobs/{jid}")
            if st.json()["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.2)
        wait_jobs_idle()


def test_jobs_busy_message(client, wait_jobs_idle):
    import time

    from api import jobs as jobs_mod

    wait_jobs_idle()

    def sleeper(rec):
        time.sleep(1.5)
        rec.message = "slept"

    jobs_mod.submit_job("sleep", sleeper)
    r2 = client.post(
        "/v1/jobs/train",
        json={"data_path": "data/raw/ehr_data.csv", "model_kind": "logreg"},
    )
    assert r2.status_code == 409
    assert "Another job" in str(r2.json()["detail"])
    wait_jobs_idle(timeout=20)
