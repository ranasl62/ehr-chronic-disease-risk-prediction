"""PNG presence validation for report figures (SHAP / calibration stubs)."""

from __future__ import annotations

from pathlib import Path

from utils.report_images import is_valid_report_png


def _minimal_png(width: int = 1, height: int = 1) -> bytes:
    from utils.report_images import minimal_png_bytes

    return minimal_png_bytes(width, height)


def test_is_valid_report_png_rejects_magic_only_stub(tmp_path: Path):
    stub = tmp_path / "shap_summary.png"
    stub.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert stub.stat().st_size == 8
    assert is_valid_report_png(stub) is False


def test_is_valid_report_png_rejects_missing_and_non_png(tmp_path: Path):
    assert is_valid_report_png(tmp_path / "missing.png") is False
    other = tmp_path / "x.png"
    other.write_bytes(b"not-a-png" + b"\x00" * 80)
    assert is_valid_report_png(other) is False


def test_is_valid_report_png_accepts_ihdr_with_dims(tmp_path: Path):
    good = tmp_path / "shap_summary.png"
    good.write_bytes(_minimal_png(8, 8))
    assert is_valid_report_png(good) is True


def test_workspace_status_ignores_corrupt_shap(client, tmp_path, monkeypatch):
    from utils import config

    reports = tmp_path / "reports"
    reports.mkdir()
    stub = reports / "shap_summary.png"
    stub.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    # researcher_routes imported REPORTS_DIR at module load — patch there too
    import api.researcher_routes as rr

    monkeypatch.setattr(rr, "REPORTS_DIR", reports)

    r = client.get("/v1/workspace/status")
    assert r.status_code == 200
    js = r.json()
    assert js["shap_present"] is False
    assert js["checklist"]["shap_available"] is False


def test_reports_summary_omits_corrupt_png(client, tmp_path, monkeypatch):
    import api.researcher_routes as rr

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cal = reports / "calibration_holdout.png"
    cal.write_bytes(_minimal_png(4, 4))
    monkeypatch.setattr(rr, "REPORTS_DIR", reports)

    r = client.get("/v1/reports/summary")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()["files"]}
    assert "shap_summary.png" not in names
    assert "calibration_holdout.png" in names


def test_trust_flags_reject_corrupt_shap(tmp_path: Path):
    from openhealth.trust_pack import build_trust_pack, trust_flags_from_dir

    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    (run_dir / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    flags = trust_flags_from_dir(run_dir)
    assert flags["has_shap"] is False
    pack = build_trust_pack("run_x", run_dir)
    assert pack["flags"]["has_shap"] is False
    assert "shap_summary.png" not in pack["artifacts"]
