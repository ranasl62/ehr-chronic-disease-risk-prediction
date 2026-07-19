"""Shared pytest fixtures for API and openhealth tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from utils.config import PROJECT_ROOT


@pytest.fixture(autouse=True)
def _reset_workspace_config():
    """Avoid polluted horizon/persona from prior tests breaking demo train."""
    try:
        from openhealth.config_store import default_config, save_config

        save_config(default_config())
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _api_teardown():
    yield
    try:
        from api.main import app, get_artifact

        app.dependency_overrides.clear()
        get_artifact.cache_clear()
    except Exception:
        pass


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """Minimal longitudinal CSV suitable for smoke train/map tests."""
    rows = []
    for pid in (1, 2, 3, 4, 5):
        for i, day in enumerate(("2023-01-01", "2023-02-01", "2023-06-01")):
            rows.append(
                {
                    "patient_id": pid,
                    "timestamp": day,
                    "glucose": 100 + pid * 5 + i,
                    "blood_pressure": 120 + pid,
                    "age": 40 + pid,
                    "label": 1 if pid > 3 else 0,
                }
            )
    path = tmp_path / "tiny_cohort.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    """CSV with alias column names requiring schema map."""
    df = pd.DataFrame(
        [
            {
                "member_id": 1,
                "service_date": "2023-01-01",
                "blood_glucose": 110,
                "outcome": 0,
                "age_years": 45,
            },
            {
                "member_id": 1,
                "service_date": "2023-06-01",
                "blood_glucose": 118,
                "outcome": 0,
                "age_years": 45,
            },
            {
                "member_id": 2,
                "service_date": "2023-01-01",
                "blood_glucose": 150,
                "outcome": 1,
                "age_years": 62,
            },
        ]
    )
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def wait_jobs_idle():
    """Block until no researcher job is queued/running (max ~15s)."""
    import time

    from api.jobs import list_recent_jobs

    def _wait(timeout: float = 15.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            busy = [j for j in list_recent_jobs(30) if j["status"] in ("queued", "running")]
            if not busy:
                return
            time.sleep(0.2)

    _wait()
    yield _wait
