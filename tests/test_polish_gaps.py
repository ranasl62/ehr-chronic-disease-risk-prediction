"""Task-aware health gates, corrupt PNG HTTP 404, ZIP exclusion."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from utils.report_images import is_valid_report_png, minimal_png_bytes


def test_train_task_aware_health_blocks_tiny_demo_horizon(client):
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/demo/ehr_data.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "task_id": "horizon_detection_30d",
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail.get("message")
    blockers = " ".join(detail.get("blockers") or [])
    assert "index_time" in blockers.lower() or "index" in blockers.lower()
    assert "paper_synthetic" in str(detail.get("hint", "")).lower() or "hint" in detail


def test_compare_task_aware_health_blocks_tiny_demo_horizon(client):
    r = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/demo/ehr_data.csv",
            "data_format": "longitudinal",
            "task_id": "horizon_detection_30d",
            "models": ["logreg"],
        },
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail.get("blockers")


def test_reports_file_404_on_corrupt_png(client, tmp_path, monkeypatch):
    import api.researcher_routes as rr

    reports = tmp_path / "reports"
    reports.mkdir()
    stub = reports / "shap_summary.png"
    stub.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 80)
    assert is_valid_report_png(stub) is False
    monkeypatch.setattr(rr, "REPORTS_DIR", reports)

    r = client.get("/v1/reports/file/shap_summary.png")
    assert r.status_code == 404
    assert "corrupt" in r.json()["detail"].lower() or "regenerate" in r.json()["detail"].lower()


def test_results_zip_excludes_corrupt_png(tmp_path, monkeypatch):
    import api.data_io as data_io

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 80)
    good = reports / "calibration_holdout.png"
    good.write_bytes(minimal_png_bytes(8, 8))
    (reports / "evaluation_report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data_io, "REPORTS_DIR", reports)

    blob = data_io.build_results_zip(run_id=None)
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        names = zf.namelist()
    assert "reports/calibration_holdout.png" in names
    assert "reports/shap_summary.png" not in names
    assert "reports/evaluation_report.json" in names


def test_is_valid_report_png_rejects_ihdr_without_iend(tmp_path: Path):
    forged = tmp_path / "fake.png"
    # IHDR-looking header padded to size, no IEND / bad CRC.
    forged.write_bytes(minimal_png_bytes(8, 8)[:40] + b"\x00" * 40)
    assert is_valid_report_png(forged) is False


def test_promote_skips_and_clears_corrupt_shared_png(tmp_path, monkeypatch):
    from openhealth import runs as runs_mod
    from openhealth.runs import promote_run
    from utils.report_images import is_valid_report_png

    run_dir = tmp_path / "runs" / "run_z"
    run_dir.mkdir(parents=True)
    (run_dir / "model.pkl").write_bytes(b"pkl")
    (run_dir / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 80)

    shared = tmp_path / "reports"
    shared.mkdir()
    stale = shared / "shap_summary.png"
    stale.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 80)

    monkeypatch.setattr(runs_mod, "run_path", lambda rid: run_dir)
    monkeypatch.setattr(runs_mod, "MODEL_PATH", shared / "model.pkl")
    monkeypatch.setattr(runs_mod, "REPORTS_DIR", shared)
    monkeypatch.setattr(
        "openhealth.trust_pack.write_trust_pack",
        lambda *a, **k: shared / "trust_pack.json",
    )
    monkeypatch.setattr("openhealth.events.emit", lambda *a, **k: None)

    out = promote_run("run_z")
    assert out["run_id"] == "run_z"
    assert not stale.exists()
    promoted = shared / "shap_summary.png"
    assert not promoted.is_file() or not is_valid_report_png(promoted)
