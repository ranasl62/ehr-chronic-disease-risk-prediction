import pytest


@pytest.fixture(autouse=True)
def _api_teardown():
    yield
    try:
        from api.main import app, get_artifact

        app.dependency_overrides.clear()
        get_artifact.cache_clear()
    except Exception:
        pass
