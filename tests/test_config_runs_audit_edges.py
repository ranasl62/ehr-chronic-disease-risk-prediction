"""Config store, events, runs, clinical audit edge cases."""

import pytest

from openhealth.clinical_audit import append_audit, recent_audit
from openhealth.config_store import (
    default_config,
    effective_train_params,
    load_config,
    save_config,
    validate_config,
)
from openhealth.events import clear_events, emit, list_events
from openhealth.runs import ensure_run, new_run_id, promote_run, run_path, write_run_meta


def test_load_config_missing_file_returns_defaults(tmp_path):
    cfg = load_config(path=tmp_path / "nope.yaml")
    assert cfg["persona"] == "researcher"
    assert cfg["model_kind"] == "logreg"


def test_validate_rejects_bad_compare_model():
    cfg = default_config()
    cfg["compare_models"] = ["logreg", "lstm"]
    with pytest.raises(ValueError, match="compare model"):
        validate_config(cfg)


def test_validate_rejects_bad_model_kind():
    cfg = default_config()
    cfg["model_kind"] = "lstm"
    with pytest.raises(ValueError, match="model_kind"):
        validate_config(cfg)


def test_effective_train_params_keys():
    cfg = default_config()
    cfg["horizon_days"] = 365
    cfg["calibrate"] = True
    p = effective_train_params(cfg)
    assert p["horizon_days"] == 365
    assert p["calibrate"] is True
    assert "windows_days" in p


def test_config_api_rejects_bad_persona(client):
    r = client.put("/v1/workspace/config", json={"persona": "superuser"})
    assert r.status_code in (400, 422)


def test_task_upsert_writes_yaml(client):
    r = client.post(
        "/v1/tasks",
        json={
            "id": "unit_task_tmp",
            "name": "Unit Task",
            "target_column": "label",
            "horizon_days": 30,
            "suggested_path": "data/raw/ehr_data.csv",
        },
    )
    assert r.status_code == 200
    from utils.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "tasks" / "unit_task_tmp.yaml").is_file()


def test_events_limit_and_clear():
    clear_events()
    for i in range(5):
        emit("t", f"m{i}", i=i)
    assert len(list_events(limit=2)) == 2
    clear_events()
    assert list_events() == []


def test_events_api_limit(client):
    clear_events()
    emit("api_test", "one")
    r = client.get("/v1/events", params={"limit": 1})
    assert r.status_code == 200
    assert len(r.json()["events"]) <= 1


def test_run_path_rejects_traversal():
    with pytest.raises(ValueError):
        run_path("../evil")
    with pytest.raises(ValueError):
        run_path("a/b")


def test_promote_missing_model_raises():
    rid = new_run_id("empty")
    ensure_run(rid)
    write_run_meta(rid, {"kind": "empty"})
    with pytest.raises(FileNotFoundError):
        promote_run(rid)


def test_clinical_audit_roundtrip(tmp_path, monkeypatch):
    from openhealth import clinical_audit as ca

    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(ca, "AUDIT_PATH", path)
    append_audit("predict", {"n": 1, "run_id": "r1"})
    append_audit("predict", {"n": 2, "run_id": "r2"})
    rows = recent_audit(limit=10)
    assert len(rows) == 2
    assert rows[-1]["run_id"] == "r2"


def test_worklist_audit_api(client):
    r = client.get("/v1/worklist/audit", params={"limit": 5})
    assert r.status_code == 200
    assert "audit" in r.json()


def test_worklist_too_many_rows(client):
    rows = [{"x": 1.0}] * 201
    r = client.post("/v1/worklist/predict", json={"rows": rows})
    assert r.status_code == 400
