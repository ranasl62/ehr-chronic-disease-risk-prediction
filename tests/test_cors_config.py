"""CORS_ORIGINS env parsing."""

from api.cors_config import parse_cors_origins


def test_cors_defaults_when_unset_or_empty():
    defaults = parse_cors_origins("")
    assert "http://localhost:8080" in defaults
    assert "http://127.0.0.1:4200" in defaults
    assert parse_cors_origins("   ") == defaults


def test_cors_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://ehr-risk-framework.larucare.com, http://localhost:8080/",
    )
    origins = parse_cors_origins()
    assert origins == [
        "https://ehr-risk-framework.larucare.com",
        "http://localhost:8080/",
    ]


def test_cors_explicit_raw_overrides_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://ignored.example")
    assert parse_cors_origins("https://ui.example") == ["https://ui.example"]
