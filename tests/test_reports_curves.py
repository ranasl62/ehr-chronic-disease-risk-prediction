"""API coverage for /v1/reports/curves."""

from __future__ import annotations

import json

from utils.config import REPORTS_DIR


def test_reports_curves_missing(client):
    ep = REPORTS_DIR / "evaluation_report.json"
    backup = ep.read_text(encoding="utf-8") if ep.is_file() else None
    try:
        if ep.is_file():
            ep.unlink()
        r = client.get("/v1/reports/curves")
        assert r.status_code == 404
    finally:
        if backup is not None:
            ep.write_text(backup, encoding="utf-8")


def test_reports_curves_and_summary(client, tmp_path):
    curves = {
        "roc": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]},
        "pr": {"precision": [1.0, 0.5], "recall": [0.0, 1.0], "thresholds": [1.0]},
        "calibration": {
            "bin_mid": [0.25, 0.75],
            "frac_positive": [0.2, 0.8],
            "mean_predicted": [0.25, 0.75],
            "counts": [2, 2],
        },
        "notes": [],
    }
    payload = {
        "metrics": {"roc_auc": 0.8, "pr_auc": 0.7},
        "curves": curves,
        "bootstrap_cis": {
            "n_boot": 50,
            "alpha": 0.05,
            "roc_auc_ci": [0.6, 0.9],
            "pr_auc_ci": [0.5, 0.85],
            "note": "percentile_bootstrap",
        },
        "quality_note": "ROC-AUC bootstrap 95% CI ≈ [0.600, 0.900]",
    }
    ep = REPORTS_DIR / "evaluation_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = ep.read_text(encoding="utf-8") if ep.is_file() else None
    try:
        ep.write_text(json.dumps(payload), encoding="utf-8")
        r = client.get("/v1/reports/curves")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["curves"]["roc"]["fpr"] == [0.0, 1.0]
        assert body["quality_note"]
        s = client.get("/v1/reports/summary")
        assert s.status_code == 200
        assert s.json().get("curves")
        assert s.json().get("bootstrap_cis")
    finally:
        if backup is not None:
            ep.write_text(backup, encoding="utf-8")
        elif ep.is_file():
            ep.unlink()


def test_reports_curves_no_curves_key(client):
    ep = REPORTS_DIR / "evaluation_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = ep.read_text(encoding="utf-8") if ep.is_file() else None
    try:
        ep.write_text(json.dumps({"metrics": {"roc_auc": 0.5}}), encoding="utf-8")
        r = client.get("/v1/reports/curves")
        assert r.status_code == 404
        assert "curves" in (r.json().get("detail") or "").lower()
    finally:
        if backup is not None:
            ep.write_text(backup, encoding="utf-8")
        elif ep.is_file():
            ep.unlink()


def test_reports_curves_invalid_json(client):
    ep = REPORTS_DIR / "evaluation_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = ep.read_text(encoding="utf-8") if ep.is_file() else None
    try:
        ep.write_text("{not-json", encoding="utf-8")
        r = client.get("/v1/reports/curves")
        assert r.status_code == 500
    finally:
        if backup is not None:
            ep.write_text(backup, encoding="utf-8")
        elif ep.is_file():
            ep.unlink()


def test_reports_curves_bad_run_id(client):
    r = client.get("/v1/reports/curves", params={"run_id": "../x"})
    assert r.status_code == 400


def test_reports_curves_missing_run_dir(client):
    r = client.get("/v1/reports/curves", params={"run_id": "no_such_run_zzz"})
    assert r.status_code == 404


def test_list_datasets_stat_oserror(client, monkeypatch, tmp_path):
    from pathlib import Path
    import api.researcher_routes as rr

    link = rr.UPLOADS_DIR / "stat_fail.csv"
    rr.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    link.write_text("a,b\n1,2\n", encoding="utf-8")
    real_stat = Path.stat

    def boom(self, *a, **k):
        if self.name == "stat_fail.csv":
            raise OSError("boom")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", boom)
    try:
        r = client.get("/v1/datasets")
        assert r.status_code == 200
        row = next(d for d in r.json()["datasets"] if d["id"] == "upload:stat_fail.csv")
        assert row["exists"] is False
        assert row["bytes"] == 0
    finally:
        link.unlink(missing_ok=True)

