"""LIMITATIONS.md surface must exist (full content filled in build; sections required)."""

from pathlib import Path

from utils.config import PROJECT_ROOT


def test_limitations_md_exists_with_core_sections():
    path = PROJECT_ROOT / "LIMITATIONS.md"
    assert path.is_file(), "LIMITATIONS.md must exist at repo root"
    text = path.read_text(encoding="utf-8")
    for section in (
        "Clinical",
        "Data",
        "Modeling",
        "Evaluation",
        "Platform",
        "Customization",
        "Forbidden",
    ):
        assert section in text, f"LIMITATIONS.md missing section hint: {section}"


def test_conftest_fixtures_importable(tiny_csv, messy_csv, client, project_root):
    assert tiny_csv.is_file()
    assert messy_csv.is_file()
    assert project_root.is_dir()
    r = client.get("/health")
    assert r.status_code == 200
