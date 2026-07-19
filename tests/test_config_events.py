"""Config store + API + events."""

from openhealth.config_store import default_config, load_config, save_config, validate_config
from openhealth.events import clear_events, emit, list_events


def test_config_round_trip(tmp_path):
    path = tmp_path / "workspace.yaml"
    cfg = default_config()
    cfg["horizon_days"] = 180
    cfg["persona"] = "researcher"
    save_config(cfg, path=path)
    loaded = load_config(path=path)
    assert loaded["horizon_days"] == 180


def test_config_rejects_bad_persona():
    cfg = default_config()
    cfg["persona"] = "admin"
    try:
        validate_config(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_config_api(client):
    r = client.get("/v1/workspace/config")
    assert r.status_code == 200
    assert "config" in r.json()
    r2 = client.put("/v1/workspace/config", json={"horizon_days": 90, "persona": "researcher"})
    assert r2.status_code == 200
    assert r2.json()["config"]["horizon_days"] == 90


def test_task_upsert_rejects_traversal(client):
    r = client.post(
        "/v1/tasks",
        json={"id": "../evil", "name": "x"},
    )
    assert r.status_code in (400, 422)


def test_events_api(client):
    clear_events()
    emit("test", "hello")
    r = client.get("/v1/events", params={"limit": 10})
    assert r.status_code == 200
    assert any(e.get("kind") == "test" for e in r.json()["events"])
