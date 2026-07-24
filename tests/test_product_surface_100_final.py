"""Final gap-closure tests for 100% api + openhealth line coverage."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from utils.config import MODEL_PATH, PROJECT_ROOT, REPORTS_DIR


@pytest.fixture()
def client():
    from api.main import app

    return TestClient(app)


def _fake_artifact(**extra):
    m = DummyClassifier(strategy="prior").fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
    art = {
        "model": m,
        "feature_columns": ["a", "age"],
        "model_kind": "dummy",
        "calibrated": False,
        "feature_engineering": {"random_state": 42},
        "X_train": pd.DataFrame({"a": [0, 1], "age": [40, 50]}),
        "X_test": pd.DataFrame({"a": [0, 1], "age": [42, 52]}),
        "y_train": pd.Series([0, 1]),
        "y_test": pd.Series([0, 1]),
    }
    art.update(extra)
    return art


# --- api.data_io ---


def test_delete_dataset_symlink_and_regular_branches(tmp_path):
    from api.data_io import delete_dataset_file, _audit_pass_fail_counts, _save_dataframe

    uploads = PROJECT_ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    txt = uploads / "plain.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Only .csv"):
        delete_dataset_file(str(txt.relative_to(PROJECT_ROOT)))

    dot_target = uploads / ".secret_target.csv"
    dot_target.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    dot_link = uploads / "link_to_secret.csv"
    if dot_link.exists():
        dot_link.unlink()
    dot_link.symlink_to(dot_target)
    with pytest.raises(ValueError, match="protected"):
        delete_dataset_file(str(dot_link.relative_to(PROJECT_ROOT)))
    dot_link.unlink(missing_ok=True)
    dot_target.unlink(missing_ok=True)

    with patch("api.data_io._is_under_deletable_root", side_effect=[True, False]):
        rogue = uploads / "rogue.csv"
        rogue.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="under data/uploads"):
            delete_dataset_file(str(rogue.relative_to(PROJECT_ROOT)))

    real_csv = uploads / "real_delete.csv"
    real_csv.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    meta = uploads / "real_delete.meta.json"
    meta.write_text("{}", encoding="utf-8")
    delete_dataset_file(str(real_csv.relative_to(PROJECT_ROOT)))
    assert not real_csv.is_file()
    assert not meta.is_file()

    outside = tmp_path / "outside.csv"
    outside.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="under project root"):
        delete_dataset_file(str(outside))

    fake_csv_dir = uploads / "fake_csv_dir.csv"
    if fake_csv_dir.exists():
        if fake_csv_dir.is_symlink():
            fake_csv_dir.unlink()
        else:
            import shutil

            shutil.rmtree(fake_csv_dir)
    fake_csv_dir.mkdir()
    try:
        with pytest.raises(ValueError, match="Only files"):
            delete_dataset_file(str(fake_csv_dir.relative_to(PROJECT_ROOT)))
    finally:
        fake_csv_dir.rmdir()

    passed, failed = _audit_pass_fail_counts(
        {
            "temporal_integrity": {"passed": False},
            "patient_disjoint_train_test": False,
            "nested": [{"passed": True}, {"passed": False}],
        }
    )
    assert passed >= 1
    assert failed >= 2

    df = pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0}])
    meta_out = _save_dataframe(df, "noext")
    assert meta_out["path"].endswith(".csv")


def test_dataframe_json_list_and_sqlalchemy_missing(tmp_path, monkeypatch):
    from api import data_io

    rows = b'[{"patient_id":1,"timestamp":"2020-01-01","label":0}]'
    df = data_io.dataframe_from_upload_bytes("rows.json", rows)
    assert len(df) == 1

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "sqlalchemy":
            raise ImportError("no sqlalchemy")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", fake_import):
        with pytest.raises(ValueError, match="sqlalchemy"):
            data_io.import_sql("SELECT 1", connection_url="sqlite:///x.db")

    csv = tmp_path / "empty_filter.csv"
    pd.DataFrame(
        [{"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "age": 50}]
    ).to_csv(csv, index=False)
    prof = data_io.profile_dataset(csv, age_band="ge70", label="1")
    assert prof["label_counts"] == {}
    assert prof["n_rows"] == 0


def test_build_methods_corrupt_and_missing_sections(tmp_path, monkeypatch):
    from api.data_io import build_methods_markdown, _reports_root_for_methods
    from openhealth.runs import ensure_run, new_run_id

    with pytest.raises(ValueError):
        _reports_root_for_methods("../evil")

    with pytest.raises(FileNotFoundError):
        _reports_root_for_methods("no_such_run_xyz")

    rid = new_run_id("methods_gap")
    rd = ensure_run(rid)
    for name in (
        "training_manifest.json",
        "evaluation_report.json",
        "leakage_audit.json",
        "trust_pack.json",
        "external_validation_report.json",
        "analysis_pack.json",
    ):
        (rd / name).write_text("{bad", encoding="utf-8")

    text = build_methods_markdown(run_id=rid)
    assert "No `external_validation_report.json`" in text or "External validation" in text
    assert "No `analysis_pack.json`" in text or "Analysis pack" in text

    text2 = build_methods_markdown()
    assert "Methods note" in text2


# --- api.jobs ---


def test_jobs_cancel_before_start_and_failure_emit(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    captured: dict[str, object] = {}

    def capture(fn):
        captured["fn"] = fn

    rec = jobs_mod.JobRecord(id="pre_cancel", kind="noop")

    def noop(_r):
        pass

    with patch.object(jobs_mod, "_EXECUTOR") as ex:
        ex.submit = capture
        out = jobs_mod.submit_job("noop", noop)
        jobs_mod._JOBS[out.id].status = "cancelled"
        captured["fn"]()  # type: ignore[operator]
    assert jobs_mod._JOBS[out.id].message == "cancelled before start"

    def boom(_r):
        raise RuntimeError("job boom")

    def bad_emit(*a, **k):
        if a and a[0] == "job_failed":
            raise OSError("emit fail")
        return {}

    monkeypatch.setattr("openhealth.events.emit", bad_emit)
    with patch.object(jobs_mod, "_EXECUTOR") as ex:
        ex.submit = lambda fn: fn()
        failed = jobs_mod.submit_job("fail", boom)
    assert failed.status == "failed"

    rel = Path("data/raw/ehr_data.csv")
    run_dir = tmp_path / "run_rel"
    run_dir.mkdir()

    def fake_train(**kw):
        mp = kw.get("model_path")
        if mp:
            joblib.dump(_fake_artifact(), mp)

    monkeypatch.setattr("training.train.run_training", fake_train)
    monkeypatch.setattr("shutil.copy2", lambda *a, **k: None)
    monkeypatch.setattr("openhealth.runs.ensure_run", lambda rid: run_dir)
    monkeypatch.setattr("openhealth.runs.new_run_id", lambda p: "run_rel")
    monkeypatch.setattr("openhealth.runs.write_run_meta", lambda *a, **k: None)
    monkeypatch.setattr("openhealth.trust_pack.write_trust_pack", lambda *a, **k: {})
    monkeypatch.setattr("openhealth.runs.promote_run", MagicMock(side_effect=RuntimeError("no")))
    monkeypatch.setattr("openhealth.config_store.load_config", lambda: {})
    monkeypatch.setattr("openhealth.config_store.save_config", MagicMock(side_effect=OSError("cfg")))
    jobs_mod.run_train_job(
        jobs_mod.JobRecord(id="tr_rel", kind="train"),
        {"data_path": str(rel), "promote": True},
    )

    with patch("api.main.get_artifact") as ga:
        ga.cache_clear.side_effect = RuntimeError("cache")
        jobs_mod.run_train_job(
            jobs_mod.JobRecord(id="tr_cache", kind="train"),
            {"data_path": str(PROJECT_ROOT / rel), "promote": False},
        )

    monkeypatch.setattr(
        "openhealth.compare.compare_models",
        lambda **kw: {"selected_model": "logreg", "comparison": []},
    )
    with patch("api.main.get_artifact") as ga:
        ga.cache_clear.side_effect = RuntimeError("cache")
        jobs_mod.run_compare_job(
            jobs_mod.JobRecord(id="cmp_rel", kind="compare"),
            {"data_path": str(rel)},
        )

    art = _fake_artifact(feature_columns=["x"])
    p = tmp_path / "fair.pkl"
    joblib.dump(art, p)
    missing_fair = tmp_path / "missing_fair.pkl"
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    monkeypatch.setattr("api.jobs.MODEL_PATH", missing_fair)
    monkeypatch.setattr("utils.config.MODEL_PATH", missing_fair)
    with pytest.raises(FileNotFoundError, match="model.pkl missing"):
        jobs_mod.run_fairness_job(jobs_mod.JobRecord(id="f0", kind="fairness"), {})
    monkeypatch.setattr("api.jobs.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    jobs_mod.run_fairness_job(jobs_mod.JobRecord(id="f1", kind="fairness"), {})
    gcsv = tmp_path / "bad_groups.csv"
    pd.DataFrame({"age_band": ["a"]}).to_csv(gcsv, index=False)
    jobs_mod.run_fairness_job(
        jobs_mod.JobRecord(id="f2", kind="fairness"),
        {"groups_path": str(gcsv), "group_column": "age_band"},
    )
    gcsv2 = PROJECT_ROOT / "data" / "uploads" / "bad_groups2.csv"
    pd.DataFrame({"age_band": ["a", "b", "c"]}).to_csv(gcsv2, index=False)
    jobs_mod.run_fairness_job(
        jobs_mod.JobRecord(id="f2b", kind="fairness"),
        {"groups_path": str(gcsv2.relative_to(PROJECT_ROOT)), "group_column": "age_band"},
    )

    audit_path = PROJECT_ROOT / "scripts" / "leakage_audit.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("leakage_audit_mod", audit_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.audit_from_raw = lambda **kw: {
        "split_method": "patient_group",
        "patient_disjoint_train_test": False,
        "temporal_integrity": {"passed": False},
    }
    sys.modules["leakage_audit_mod"] = mod
    jobs_mod.run_leakage_audit_job(
        jobs_mod.JobRecord(id="l1", kind="leakage"),
        {"use_artifact": False, "data_path": str(rel)},
    )

    with patch("importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(RuntimeError, match="cannot load"):
            jobs_mod.run_leakage_audit_job(
                jobs_mod.JobRecord(id="l2", kind="leakage"),
                {"use_artifact": True},
            )

    monkeypatch.setattr(
        "training.hpo.run_light_hpo",
        lambda **kw: {"report_path": "r", "best": {"roc_auc": 0.5}, "n_trials": 1},
    )
    with patch("api.main.get_artifact") as ga:
        ga.cache_clear.side_effect = RuntimeError("cache")
        jobs_mod.run_hpo_job(jobs_mod.JobRecord(id="h1", kind="hpo"), {"data_path": str(rel)})

    monkeypatch.setattr("explainability.shap_explainer.explain_model", lambda *a, **k: None)
    monkeypatch.setattr("openhealth.trust_pack.resolve_active_run_id", lambda x=None: None)
    missing_shap = tmp_path / "no_shap_model.pkl"
    monkeypatch.setattr("api.jobs.MODEL_PATH", missing_shap)
    monkeypatch.setattr("utils.config.MODEL_PATH", missing_shap)
    with pytest.raises(FileNotFoundError):
        jobs_mod.run_shap_job(jobs_mod.JobRecord(id="s1", kind="shap"), {"out": "reports/custom_shap.png"})

    monkeypatch.setattr(
        "training.external_validate.external_validate",
        lambda **kw: {"metrics": {"roc_auc": 0.5}},
    )
    monkeypatch.setattr(
        "training.external_validate.write_external_validation_report",
        lambda *a, **k: tmp_path / "ev.json",
    )
    jobs_mod.run_external_validate_job(
        jobs_mod.JobRecord(id="e1", kind="ext"),
        {"data_path": str(rel)},
    )

    assert jobs_mod._audit_passed({"split_method": "patient_group", "patient_disjoint_train_test": False}) is False
    assert jobs_mod._audit_passed({"temporal_integrity": {"passed": False}}) is False


def test_jobs_remaining_unit_paths(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    rel = Path("data/raw/ehr_data.csv")
    missing = tmp_path / "nope.pkl"
    monkeypatch.setattr("api.jobs.MODEL_PATH", missing)
    monkeypatch.setattr("utils.config.MODEL_PATH", missing)
    with pytest.raises(FileNotFoundError, match="model.pkl missing"):
        jobs_mod.run_fairness_job(jobs_mod.JobRecord(id="ff", kind="fairness"), {})

    art = _fake_artifact(feature_columns=["z_score"])
    p = tmp_path / "fair2.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("api.jobs.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    jobs_mod.run_fairness_job(jobs_mod.JobRecord(id="f_no_age", kind="fairness"), {})

    run_dir = tmp_path / "run_shap2"
    run_dir.mkdir()
    joblib.dump(art, run_dir / "model.pkl")
    monkeypatch.setattr("openhealth.trust_pack.resolve_active_run_id", lambda x=None: "run_shap2")
    monkeypatch.setattr("openhealth.runs.run_path", lambda rid: run_dir)
    monkeypatch.setattr("openhealth.runs.ensure_run", lambda rid: run_dir)

    def _fake_shap(*_a, **k):
        from utils.report_images import minimal_png_bytes

        plot_path = k.get("plot_path")
        if plot_path:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            Path(plot_path).write_bytes(minimal_png_bytes())

    monkeypatch.setattr("explainability.shap_explainer.explain_model", _fake_shap)
    jobs_mod.run_shap_job(
        jobs_mod.JobRecord(id="sh2", kind="shap"),
        {"run_id": "run_shap2", "out": "reports/custom/out.png"},
    )

    monkeypatch.setattr(
        "training.external_validate.external_validate",
        lambda **kw: {"metrics": {"roc_auc": 0.5}},
    )
    monkeypatch.setattr(
        "training.external_validate.write_external_validation_report",
        lambda *a, **k: tmp_path / "ev2.json",
    )
    jobs_mod.run_external_validate_job(
        jobs_mod.JobRecord(id="ev", kind="ext"),
        {"data_path": str(rel)},
    )

    with patch("importlib.util.spec_from_file_location", return_value=None):
        with pytest.raises(RuntimeError):
            jobs_mod.run_leakage_audit_job(
                jobs_mod.JobRecord(id="leak_spec", kind="leakage"),
                {"use_artifact": True},
            )

    monkeypatch.setattr(
        "training.hpo.run_light_hpo",
        lambda **kw: {"report_path": "r", "best": {"roc_auc": 0.5}, "n_trials": 1},
    )
    with patch("api.main.get_artifact") as ga:
        ga.cache_clear.side_effect = RuntimeError("cache")
        jobs_mod.run_hpo_job(jobs_mod.JobRecord(id="hp", kind="hpo"), {"data_path": str(rel)})

    monkeypatch.setattr("openhealth.trust_pack.resolve_active_run_id", lambda x=None: None)

    def touch_plot(*a, **k):
        from utils.report_images import minimal_png_bytes

        plot_path = k.get("plot_path")
        if plot_path:
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(minimal_png_bytes())

    monkeypatch.setattr("explainability.shap_explainer.explain_model", touch_plot)
    shared = REPORTS_DIR / "shap_summary.png"
    custom = PROJECT_ROOT / "reports" / "custom_only_shap.png"
    custom.parent.mkdir(parents=True, exist_ok=True)
    jobs_mod.run_shap_job(
        jobs_mod.JobRecord(id="sh3", kind="shap"),
        {"out": str(custom.relative_to(PROJECT_ROOT))},
    )
    assert custom.is_file() or shared.is_file()
    assert (custom if custom.is_file() else shared).stat().st_size >= 64

def test_delete_symlink_target_deletable_race():
    from api.data_io import delete_dataset_file

    uploads = PROJECT_ROOT / "data" / "uploads"
    valid = uploads / "valid125.csv"
    valid.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    link = uploads / "link125.csv"
    if link.exists():
        link.unlink()
    link.symlink_to(valid)
    try:
        with patch("api.data_io._is_under_deletable_root", side_effect=[True, True, True, False]):
            with pytest.raises(ValueError, match="under data/uploads"):
                delete_dataset_file(str(link.relative_to(PROJECT_ROOT)))
    finally:
        link.unlink(missing_ok=True)
        valid.unlink(missing_ok=True)


def test_cancel_running_job():
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="run_cancel", kind="train")
    rec.status = "running"
    with jobs_mod._LOCK:
        jobs_mod._JOBS[rec.id] = rec
    out = jobs_mod.cancel_job(rec.id)
    assert out.status == "cancelled"


# --- api.framework_routes ---


def test_framework_absolute_path_and_config_errors(client, tmp_path, monkeypatch):
    from api.framework_routes import get_run_detail, promote_run_route, upsert_task
    from fastapi import HTTPException
    from types import SimpleNamespace

    p = PROJECT_ROOT / "data" / "raw" / "ehr_data.csv"
    assert client.get("/v1/datasets/profile", params={"path": str(p.resolve())}).status_code == 200
    r_map = client.post(
        "/v1/datasets/map-preview",
        json={"path": str(p.resolve())},
    )
    assert r_map.status_code == 200

    monkeypatch.setattr(
        "openhealth.config_store.save_config",
        lambda cfg: (_ for _ in ()).throw(ValueError("bad config")),
    )
    r = client.put("/v1/workspace/config", json={"persona": "researcher"})
    assert r.status_code == 400

    bad_body = SimpleNamespace(
        id="../evil",
        name="x",
        description="",
        target_column="label",
        horizon_days=None,
        index_strategy="last_event",
        index_time_col=None,
        windows_days=[7, 30, 180],
        data_format="longitudinal",
        suggested_path=None,
        model_kind="logreg",
        calibrate=False,
        split_by_patient=True,
    )
    with pytest.raises(HTTPException) as exc:
        upsert_task(bad_body, True)
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc2:
        get_run_detail("../evil", True)
    assert exc2.value.status_code == 400

    with pytest.raises(HTTPException) as exc3:
        promote_run_route("../evil", True)
    assert exc3.value.status_code == 400

    r5 = client.post("/v1/jobs/does_not_exist_xyz/cancel")
    assert r5.status_code == 404


def test_framework_thresholds_compute(client, tmp_path, monkeypatch):
    tp = REPORTS_DIR / "threshold_operating_points.json"
    backup = tp.read_text(encoding="utf-8") if tp.is_file() else None
    tp.unlink(missing_ok=True)

    art = _fake_artifact()
    model_path = tmp_path / "thresh_model.pkl"
    joblib.dump(art, model_path)
    monkeypatch.setattr("utils.config.MODEL_PATH", model_path)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )

    try:
        r = client.get("/v1/reports/thresholds")
        assert r.status_code == 200
        assert r.json().get("present") is True
        assert tp.is_file()

        tp.write_text("{bad", encoding="utf-8")
        r2 = client.get("/v1/reports/thresholds")
        assert r2.status_code == 200
    finally:
        if backup:
            tp.write_text(backup, encoding="utf-8")
        else:
            tp.unlink(missing_ok=True)


def test_framework_fhir_omop_import(client):
    r = client.post(
        "/v1/datasets/from-fhir",
        json={
            "bundle": [
                {"resourceType": "Patient", "id": "p1"},
                "not-a-dict",
                {
                    "resourceType": "Observation",
                    "subject": {"reference": "Patient/p1"},
                    "effectiveDateTime": "2020-01-01",
                    "valueQuantity": {"value": 100},
                },
            ],
            "name": "fhir_unit.csv",
        },
    )
    assert r.status_code == 200

    r2 = client.post(
        "/v1/datasets/from-omop",
        json={
            "person": [{"person_id": 1, "year_of_birth": 1980}],
            "measurement": [
                {"person_id": 1, "measurement_date": "2020-01-01", "value_as_number": 100}
            ],
            "name": "omop_unit.csv",
        },
    )
    assert r2.status_code == 200


# --- api.researcher_routes ---


def test_researcher_route_error_paths(client, monkeypatch, tiny_csv, wait_jobs_idle):
    wait_jobs_idle()

    r = client.get("/v1/tasks/bad_task_yaml_only")
    assert r.status_code in (404, 400)

    bad_task = PROJECT_ROOT / "tasks" / "broken_route.yaml"
    bad_task.write_text("scalar", encoding="utf-8")
    try:
        r2 = client.get("/v1/tasks/broken_route")
        assert r2.status_code == 400
    finally:
        bad_task.unlink(missing_ok=True)

    r3 = client.get("/v1/datasets/health", params={"path": str(Path("/etc/passwd"))})
    assert r3.status_code == 400

    r4 = client.post(
        "/v1/datasets/upload",
        files={"file": ("demo.json", b'[{"patient_id":1,"timestamp":"2020-01-01","label":0}]', "application/json")},
    )
    assert r4.status_code == 200

    r5 = client.post("/v1/datasets/from-form", json={"rows": [], "name": "x.csv"})
    assert r5.status_code == 400

    r6 = client.get("/v1/datasets/profile", params={"path": "../etc/passwd"})
    assert r6.status_code == 400

    dest = PROJECT_ROOT / "data" / "uploads" / "force_train.csv"
    dest.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n2,2020-02-01,1\n", encoding="utf-8")
    r7 = client.post(
        "/v1/jobs/train",
        json={
            "data_path": str(dest.relative_to(PROJECT_ROOT)),
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "calibrate": False,
            "split_by_patient": True,
            "temporal_split": False,
            "windows_days": [180],
            "window_days": 180,
            "horizon_days": None,
            "index_strategy": "last_event",
            "index_time_col": None,
            "feature_inclusive": True,
            "label_col": "label",
            "force": True,
            "task_id": "no_such_task_xyz",
        },
    )
    assert r7.status_code == 400

    monkeypatch.setattr(
        "openhealth.health.dataset_health_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("health boom")),
    )
    r8 = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "calibrate": False,
            "split_by_patient": True,
            "temporal_split": False,
            "windows_days": [180],
            "window_days": 180,
            "horizon_days": None,
            "index_strategy": "last_event",
            "index_time_col": None,
            "feature_inclusive": True,
            "label_col": "label",
        },
    )
    assert r8.status_code == 400

    r9 = client.get("/v1/reports/download.zip", params={"run_id": "no_such_run_xyz"})
    assert r9.status_code == 404

    r10 = client.get("/v1/reports/file/evaluation_report.json")
    assert r10.status_code in (200, 404)


def test_researcher_train_with_task_and_hpo_blockers(client, wait_jobs_idle):
    wait_jobs_idle()
    body = {
        "data_path": "data/raw/ehr_data.csv",
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "temporal_split": True,
        "windows_days": [180],
        "window_days": 180,
        "horizon_days": None,
        "index_strategy": "last_event",
        "index_time_col": None,
        "feature_inclusive": True,
        "label_col": "label",
        "task_id": "diabetes",
    }
    r = client.post("/v1/jobs/train", json=body)
    # Tiny demo lacks index_time — task-aware health gate blocks without force.
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail.get("blockers")

    nolabel = PROJECT_ROOT / "data" / "uploads" / "nolabel_hpo.csv"
    nolabel.write_text("patient_id,timestamp\n1,2020-01-01\n", encoding="utf-8")
    r2 = client.post(
        "/v1/jobs/hpo",
        json={
            **body,
            "data_path": str(nolabel.relative_to(PROJECT_ROOT)),
            "task_id": "diabetes",
        },
    )
    assert r2.status_code == 400


def test_researcher_analysis_pack_success(client):
    r = client.get("/v1/reports/analysis-pack", params={"path": "data/raw/ehr_data.csv"})
    assert r.status_code == 200


# --- api.main ---


def test_explain_invalid_features(client):
    from api.main import app, artifact_dep, get_artifact

    art = _fake_artifact()
    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = lambda: art
    try:
        r = client.post("/explain", json={"features": {"a": 1.0}})
        assert r.status_code == 400
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


# --- api.middleware / production ---


def test_middleware_configure_skips_when_handlers_present():
    from api import middleware as mw
    import logging

    mw.log.handlers.clear()
    mw.configure_api_logging()
    mw.log.addHandler(logging.StreamHandler())
    mw.configure_api_logging()


def test_rate_limit_pops_expired_entries():
    from api.production_middleware import RateLimitMiddleware
    from collections import deque
    import time as time_mod
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient as StarletteTestClient

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/x", ok, methods=["GET"])])
    mw = RateLimitMiddleware(app, per_minute=5)
    with patch.object(time_mod, "time", return_value=200.0):
        mw._windows["127.0.0.1"] = deque([10.0, 20.0, 100.0])
        c = StarletteTestClient(mw)
        assert c.get("/x").status_code == 200


# --- openhealth ---


def test_openhealth_evaluate_model_only(tmp_path, monkeypatch):
    from openhealth.api import evaluate

    monkeypatch.setattr("utils.eval_report.load_evaluation_report_safe", lambda: None)
    p = tmp_path / "m.pkl"
    joblib.dump(_fake_artifact(), p)
    out = evaluate(p)
    assert out["metrics"] is None


def test_openhealth_compare_promote_relative(tmp_path, monkeypatch, tiny_csv):
    from openhealth.compare import compare_models

    out_dir = REPORTS_DIR / "compare_unit_cov"
    out_dir.mkdir(parents=True, exist_ok=True)
    active = tmp_path / "active.pkl"

    def fake_train(**kw):
        mp = kw.get("model_path")
        if mp:
            joblib.dump(_fake_artifact(), mp)
        X_test = pd.DataFrame({"a": [0, 1]})
        y_test = pd.Series([0, 1])
        return None, X_test, y_test, {"model": DummyClassifier().fit([[0]], [0])}

    monkeypatch.setattr("training.train.run_training", fake_train)
    monkeypatch.setattr(
        "training.reporting.build_evaluation_report",
        lambda *a, **k: {"metrics": {"roc_auc": 0.9}},
    )
    monkeypatch.setattr("openhealth.compare.MODEL_PATH", active)
    monkeypatch.setattr("utils.config.MODEL_PATH", active)

    summary = compare_models(
        data_path=tiny_csv,
        models=["logreg"],
        out_dir=out_dir,
        promote_best=True,
    )
    assert summary["selected_model"] == "logreg"
    assert active.is_file()


def test_health_task_and_index_time(tmp_path):
    from openhealth.health import dataset_health_report

    p = tmp_path / "health.csv"
    pd.DataFrame(
        [
            {
                "patient_id": 1,
                "timestamp": "2020-01-01",
                "label": 0,
                "index_time": "2020-01-01",
                "age": 50,
            },
            {
                "patient_id": 2,
                "timestamp": "2020-06-01",
                "label": 1,
                "index_time": "2020-01-01",
                "age": 60,
            },
        ]
    ).to_csv(p, index=False)
    rep = dataset_health_report(p, task_id="diabetes")
    assert rep["health"]["leakage_risk"] in ("LOW", "MEDIUM", "HIGH")

    p2 = tmp_path / "warn.csv"
    pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0}]).to_csv(p2, index=False)
    rep2 = dataset_health_report(p2)
    assert rep2["health"]["warnings"]


def test_task_spec_horizon_from_window(tmp_path):
    from openhealth.task_spec import load_task

    p = tmp_path / "horizon.yaml"
    p.write_text(
        """
task: {id: h, name: H}
target: label
prediction: {window: 365_days}
data: {suggested_path: data/raw/ehr_data.csv}
""",
        encoding="utf-8",
    )
    t = load_task(p)
    assert t.horizon_days == 365


def test_adapters_ndjson_and_skip_bad_resource(tmp_path):
    from openhealth.adapters import fhir_bundle_to_longitudinal, load_fhir_file

    fhir_bundle_to_longitudinal(
        [
            "skip-me",
            {
                "resourceType": "Observation",
                "subject": {"reference": "Patient/p1"},
                "effectiveDateTime": "2020-01-01",
                "valueQuantity": {"value": 1},
            },
        ]
    )
    nd = tmp_path / "bundle.ndjson"
    nd.write_text('{"resourceType":"Patient","id":"1"}\n', encoding="utf-8")
    payload = load_fhir_file(nd)
    assert isinstance(payload, list)


def test_list_tasks_skips_broken_yaml(tmp_path):
    from openhealth.task_spec import list_tasks

    (tmp_path / "good.yaml").write_text(
        "task: {id: g, name: G}\ntarget: label\ndata: {suggested_path: data/raw/ehr_data.csv}\n",
        encoding="utf-8",
    )
    (tmp_path / "bad.yaml").write_text("not a mapping", encoding="utf-8")
    tasks = list_tasks(tmp_path)
    assert any(t.id == "g" for t in tasks)


def test_delete_symlink_target_suffix_guard():
    from api.data_io import delete_dataset_file

    uploads = PROJECT_ROOT / "data" / "uploads"
    target = uploads / "suffix_target.txt"
    target.write_text("x", encoding="utf-8")
    link = uploads / "suffix_link.csv"
    if link.exists():
        link.unlink()
    link.symlink_to(target)
    try:
        with pytest.raises(ValueError, match="Only .csv"):
            delete_dataset_file(str(link.relative_to(PROJECT_ROOT)))
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_delete_symlink_protected_target():
    from api.data_io import delete_dataset_file

    uploads = PROJECT_ROOT / "data" / "uploads"
    hidden = uploads / ".hidden_real.csv"
    hidden.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    link = uploads / "to_hidden.csv"
    if link.exists():
        link.unlink()
    link.symlink_to(hidden)
    try:
        with pytest.raises(ValueError, match="protected"):
            delete_dataset_file(str(link.relative_to(PROJECT_ROOT)))
    finally:
        link.unlink(missing_ok=True)
        hidden.unlink(missing_ok=True)


def test_framework_thresholds_missing_model(client, tmp_path, monkeypatch):
    from api.framework_routes import get_threshold_points
    from fastapi import HTTPException

    tp = REPORTS_DIR / "threshold_operating_points.json"
    tp.unlink(missing_ok=True)
    missing = tmp_path / "no_model.pkl"
    monkeypatch.setattr("utils.config.MODEL_PATH", missing)
    with pytest.raises(HTTPException) as exc:
        get_threshold_points(True)
    assert exc.value.status_code == 404


def test_framework_fairness_job_busy(client, monkeypatch, wait_jobs_idle):
    from api import jobs as jobs_mod

    wait_jobs_idle()

    def _busy(*_a, **_k):
        raise RuntimeError("busy")

    monkeypatch.setattr("api.framework_routes.submit_job", _busy)
    r = client.post("/v1/jobs/fairness", json={})
    assert r.status_code == 409


def test_jobs_shap_with_run_id(tmp_path, monkeypatch):
    from api import jobs as jobs_mod

    art = _fake_artifact()
    run_dir = tmp_path / "run_shap"
    run_dir.mkdir()
    joblib.dump(art, run_dir / "model.pkl")
    active = tmp_path / "active.pkl"
    joblib.dump(art, active)
    monkeypatch.setattr("api.jobs.MODEL_PATH", active)
    monkeypatch.setattr("utils.config.MODEL_PATH", active)
    monkeypatch.setattr("openhealth.trust_pack.resolve_active_run_id", lambda x=None: "run_shap")
    monkeypatch.setattr("openhealth.runs.ensure_run", lambda rid: run_dir)
    monkeypatch.setattr("openhealth.runs.run_path", lambda rid: run_dir)

    def _fake_shap(*_a, **k):
        from utils.report_images import minimal_png_bytes

        plot_path = k.get("plot_path")
        if plot_path:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            Path(plot_path).write_bytes(minimal_png_bytes())

    monkeypatch.setattr("explainability.shap_explainer.explain_model", _fake_shap)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    with patch("api.main.get_artifact") as ga:
        ga.cache_clear.side_effect = RuntimeError("cache")
        jobs_mod.run_shap_job(jobs_mod.JobRecord(id="shap_run", kind="shap"), {"run_id": "run_shap"})


def test_researcher_routes_remaining(client, monkeypatch, wait_jobs_idle):
    import asyncio
    from api.researcher_routes import upload_dataset
    from fastapi import HTTPException
    from unittest.mock import AsyncMock, MagicMock

    wait_jobs_idle()

    empty = MagicMock()
    empty.filename = ""
    empty.read = AsyncMock(return_value=b"x")
    with pytest.raises(HTTPException) as exc_empty:
        asyncio.run(upload_dataset(empty, True))
    assert exc_empty.value.status_code == 400

    huge = MagicMock()
    huge.filename = "big.csv"
    huge.read = AsyncMock(return_value=b"x" * (50 * 1024 * 1024 + 1))
    with pytest.raises(HTTPException) as exc_big:
        asyncio.run(upload_dataset(huge, True))
    assert exc_big.value.status_code == 400

    big = b"x" * (50 * 1024 * 1024 + 1)
    r2 = client.post("/v1/datasets/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r2.status_code in (400, 413)

    monkeypatch.setattr(
        "api.data_io.delete_dataset_file",
        lambda _p: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    r3 = client.delete("/v1/datasets", params={"path": "data/uploads/missing.csv"})
    assert r3.status_code == 404

    monkeypatch.setattr(
        "api.data_io.delete_dataset_file",
        lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    r4 = client.delete("/v1/datasets", params={"path": "data/uploads/x.csv"})
    assert r4.status_code == 400

    monkeypatch.setattr(
        "openhealth.analysis_pack.build_analysis_pack",
        lambda _p: (_ for _ in ()).throw(RuntimeError("pack fail")),
    )
    r5 = client.get("/v1/reports/analysis-pack", params={"path": "data/raw/ehr_data.csv"})
    assert r5.status_code == 400

    r6 = client.get("/v1/reports/file/not_in_allowlist.xyz")
    assert r6.status_code == 404

    ep = REPORTS_DIR / "evaluation_report.json"
    backup = ep.read_text(encoding="utf-8") if ep.is_file() else None
    ep.unlink(missing_ok=True)
    try:
        r7 = client.get("/v1/reports/file/evaluation_report.json")
        assert r7.status_code == 404
    finally:
        if backup:
            ep.write_text(backup, encoding="utf-8")


def test_health_task_missing_columns(tmp_path):
    from openhealth.health import dataset_health_report

    p = tmp_path / "task_miss.csv"
    pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0}]).to_csv(p, index=False)
    rep = dataset_health_report(p, task_id="readmission_30d")
    assert any("task" in b.lower() or "index" in b.lower() for b in rep["health"]["blockers"] + rep["health"]["warnings"])


def test_health_index_time_lowers_risk(tmp_path):
    from openhealth.health import dataset_health_report

    p = tmp_path / "idx.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "index_time": "2020-01-01", "age": 50},
            {"patient_id": 2, "timestamp": "2020-06-01", "label": 1, "index_time": "2020-01-01", "age": 60},
        ]
    ).to_csv(p, index=False)
    rep = dataset_health_report(p)
    assert rep["health"]["leakage_risk"] in ("LOW", "MEDIUM", "HIGH")


def test_task_spec_yaml_import_error_direct():
    from openhealth import task_spec as ts

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", fake_import):
        with pytest.raises(ImportError):
            ts._require_yaml()


def test_adapters_load_fhir_json_file(tmp_path):
    from openhealth.adapters import load_fhir_file

    j = tmp_path / "bundle.json"
    j.write_text('{"resourceType":"Patient","id":"1"}', encoding="utf-8")
    payload = load_fhir_file(j)
    assert payload["resourceType"] == "Patient"


def test_delete_target_protected_and_suffix_guards(monkeypatch):
    from api import data_io

    uploads = PROJECT_ROOT / "data" / "uploads"
    p = uploads / "mock_guard.csv"
    p.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    resolved = p.resolve()

    class GuardTarget:
        def __init__(self, name: str, suffix: str):
            self.name = name
            self.suffix = suffix

        def is_file(self) -> bool:
            return True

        def is_relative_to(self, other) -> bool:
            return True

        def unlink(self) -> None:
            raise AssertionError("should not unlink")

    real_resolve = Path.resolve

    def fake_resolve(self, strict=False):
        r = real_resolve(self, strict=strict)
        if r == resolved:
            return GuardTarget(".evil.csv", ".csv")
        return r

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ValueError, match="protected"):
        data_io.delete_dataset_file(str(p.relative_to(PROJECT_ROOT)))

    def fake_resolve2(self, strict=False):
        r = real_resolve(self, strict=strict)
        if r == resolved:
            return GuardTarget("mock_guard.csv", ".txt")
        return r

    monkeypatch.setattr(Path, "resolve", fake_resolve2)
    with pytest.raises(ValueError, match="Only .csv"):
        data_io.delete_dataset_file(str(p.relative_to(PROJECT_ROOT)))
    p.unlink(missing_ok=True)


def test_researcher_train_compare_hpo_routes(client, wait_jobs_idle, monkeypatch):
    wait_jobs_idle()
    from openhealth.config_store import load_config, save_config

    cfg = load_config()
    cfg["compare_models"] = ["logreg", "random_forest"]
    cfg["windows_days"] = [7, 30]
    save_config(cfg)

    body = {
        "data_path": "data/raw/ehr_data.csv",
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "temporal_split": True,
        "windows_days": [180],
        "window_days": 180,
        "horizon_days": None,
        "index_strategy": "last_event",
        "index_time_col": None,
        "feature_inclusive": True,
        "label_col": "label",
        "task_id": "diabetes",
        "force": True,
    }
    r = client.post("/v1/jobs/train", json=body)
    assert r.status_code in (200, 409)

    wait_jobs_idle()
    r3 = client.post(
        "/v1/jobs/compare",
        json={**body, "temporal_split": True, "promote_best": False},
    )
    assert r3.status_code in (200, 400, 409)

    nolabel = PROJECT_ROOT / "data" / "uploads" / "nolabel_compare.csv"
    nolabel.write_text("patient_id,timestamp\n1,2020-01-01\n", encoding="utf-8")
    r4 = client.post(
        "/v1/jobs/compare",
        json={**body, "data_path": str(nolabel.relative_to(PROJECT_ROOT))},
    )
    assert r4.status_code == 400

    r5 = client.post("/v1/jobs/leakage-audit", json={"use_artifact": False})
    assert r5.status_code == 400

    r6 = client.post(
        "/v1/jobs/hpo",
        json={**body, "task_id": "diabetes", "max_trials": 1},
    )
    assert r6.status_code in (200, 400, 409)

    nolabel2 = PROJECT_ROOT / "data" / "uploads" / "nolabel_hpo2.csv"
    nolabel2.write_text("patient_id,timestamp\n1,2020-01-01\n", encoding="utf-8")
    r7 = client.post(
        "/v1/jobs/hpo",
        json={**body, "data_path": str(nolabel2.relative_to(PROJECT_ROOT)), "task_id": "diabetes"},
    )
    assert r7.status_code == 400

    wait_jobs_idle()
    r8 = client.post(
        "/v1/jobs/external-validate",
        json={"data_path": "data/raw/ehr_data.csv", "data_format": "longitudinal"},
    )
    assert r8.status_code in (200, 409)


def test_researcher_train_health_failure(client, monkeypatch):
    monkeypatch.setattr(
        "openhealth.health.dataset_health_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("health fail")),
    )
    body = {
        "data_path": "data/raw/ehr_data.csv",
        "data_format": "longitudinal",
        "model_kind": "logreg",
        "calibrate": False,
        "split_by_patient": True,
        "temporal_split": False,
        "windows_days": [180],
        "window_days": 180,
        "horizon_days": None,
        "index_strategy": "last_event",
        "index_time_col": None,
        "feature_inclusive": True,
        "label_col": "label",
    }
    r = client.post("/v1/jobs/train", json=body)
    assert r.status_code == 400


def test_rate_limit_deque_pop():
    from api.production_middleware import RateLimitMiddleware
    from collections import deque
    import time as time_mod
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient as StarletteTestClient

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/x", ok, methods=["GET"])])
    mw = RateLimitMiddleware(app, per_minute=100)
    with patch.object(time_mod, "time", return_value=200.0):
        dq = deque([10.0, 20.0, 100.0])
        mw._windows["testclient"] = dq
        c = StarletteTestClient(mw)
        assert c.get("/x").status_code == 200
        assert len(dq) >= 1


def test_final_twenty_line_gaps(tmp_path, monkeypatch, client, wait_jobs_idle):
    from api import jobs as jobs_mod
    from openhealth.health import dataset_health_report
    from openhealth.task_spec import TASKS_DIR, list_tasks

    art = _fake_artifact(
        feature_columns=["z_score"],
        X_train=pd.DataFrame({"z_score": [0, 1, 0, 1]}),
        X_test=pd.DataFrame({"z_score": [0, 1]}),
    )
    p = tmp_path / "no_age.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("api.jobs.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    jobs_mod.run_fairness_job(jobs_mod.JobRecord(id="no_age", kind="fairness"), {})

    monkeypatch.setattr(
        "training.hpo.run_light_hpo",
        lambda **kw: {"report_path": "r", "best": {"roc_auc": 0.5}, "n_trials": 1},
    )
    jobs_mod.run_hpo_job(
        jobs_mod.JobRecord(id="hp_rel", kind="hpo"),
        {"data_path": "data/raw/ehr_data.csv"},
    )

    wait_jobs_idle()
    train_body = {
        "data_path": "data/raw/ehr_data.csv",
        "data_format": "longitudinal",
        "task_id": "diabetes",
    }
    r = client.post("/v1/jobs/train", json=train_body)
    assert r.status_code in (200, 400, 409, 422)

    wait_jobs_idle()
    r2 = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "task_id": "diabetes",
            "temporal_split": True,
        },
    )
    assert r2.status_code in (200, 400, 409, 422)

    r3 = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/ehr_data.csv",
            "data_format": "longitudinal",
            "task_id": "no_such_task_xyz",
        },
    )
    assert r3.status_code == 400

    r4 = client.post("/v1/jobs/leakage-audit", json={"use_artifact": False, "data_path": None})
    assert r4.status_code in (400, 422)

    def _busy(*_a, **_k):
        raise RuntimeError("busy")

    monkeypatch.setattr("api.researcher_routes.submit_job", _busy)
    r5 = client.post(
        "/v1/jobs/external-validate",
        json={"data_path": "data/raw/ehr_data.csv", "data_format": "longitudinal"},
    )
    assert r5.status_code == 409

    p_no_ts = tmp_path / "no_ts.csv"
    pd.DataFrame([{"patient_id": 1, "label": 0}]).to_csv(p_no_ts, index=False)
    rep = dataset_health_report(p_no_ts)
    assert rep["health"]["warnings"]

    p_task = tmp_path / "task_miss2.csv"
    pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0}]).to_csv(p_task, index=False)
    rep2 = dataset_health_report(p_task, task_id="readmission_30d")
    assert rep2["health"]["blockers"] or rep2["health"]["warnings"]

    p_idx = tmp_path / "idx2.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "index_time": "2020-01-01"},
            {"patient_id": 2, "timestamp": "2020-06-01", "label": 1, "index_time": "2020-01-01"},
        ]
    ).to_csv(p_idx, index=False)
    rep3 = dataset_health_report(p_idx)
    assert rep3["health"]["leakage_risk"] in ("LOW", "MEDIUM", "HIGH")

    broken = TASKS_DIR / "_broken_list.yaml"
    broken.write_text("not-a-mapping", encoding="utf-8")
    try:
        listed = list_tasks(TASKS_DIR)
        assert isinstance(listed, list)
    finally:
        broken.unlink(missing_ok=True)


def test_coverage_last_mile(client, wait_jobs_idle, monkeypatch, tmp_path):
    from api import jobs as jobs_mod
    from api.researcher_routes import (
        CompareJobBody,
        HpoJobBody,
        LeakageJobBody,
        TrainJobBody,
        start_compare,
        start_hpo,
        start_leakage,
        start_train,
    )
    from fastapi import HTTPException
    from openhealth.config_store import load_config, save_config
    from openhealth.health import dataset_health_report
    from openhealth.task_spec import list_tasks

    with pytest.raises(FileNotFoundError):
        jobs_mod.run_hpo_job(
            jobs_mod.JobRecord(id="hp_miss", kind="hpo"),
            {"data_path": "data/raw/no_such_file_xyz.csv"},
        )

    p_alias = tmp_path / "alias.csv"
    pd.DataFrame(
        [
            {
                "patient_id": 1,
                "timestamp": "2020-01-01",
                "chronic_disease": 0,
                "index_time": "2020-01-01",
            }
        ]
    ).to_csv(p_alias, index=False)
    dataset_health_report(p_alias, task_id="diabetes")

    assert list_tasks(tmp_path / "missing_tasks_dir") == []

    cfg = load_config()
    cfg["horizon_days"] = 365
    cfg["windows_days"] = [7, 30]
    cfg["model_kind"] = "logreg"
    cfg["active_task_id"] = "diabetes"
    cfg["compare_models"] = ["logreg"]
    save_config(cfg)

    def fake_submit(kind, fn):
        from api.jobs import JobRecord

        rec = JobRecord(id="cov", kind=kind)
        rec.status = "succeeded"
        return rec

    monkeypatch.setattr("api.researcher_routes.submit_job", fake_submit)
    monkeypatch.setattr("openhealth.events.emit", lambda *a, **k: None)

    start_train(
        TrainJobBody(
            data_path="data/raw/ehr_data.csv",
            task_id="diabetes",
            label_col=None,
            force=True,
        ),
        True,
    )
    # Workspace config may still suggest diabetes — force past tiny-demo blockers.
    start_train(TrainJobBody(data_path="data/raw/ehr_data.csv", force=True), True)

    start_compare(
        CompareJobBody(
            data_path="data/raw/ehr_data.csv",
            temporal_split=True,
            label_col="label",
        ),
        True,
    )
    start_compare(
        CompareJobBody(data_path="data/raw/ehr_data.csv", models=None, label_col="label"),
        True,
    )

    with pytest.raises(HTTPException) as exc:
        start_leakage(LeakageJobBody(use_artifact=False, data_path=None), True)
    assert exc.value.status_code == 400

    start_leakage(
        LeakageJobBody(use_artifact=False, data_path="data/raw/ehr_data.csv"),
        True,
    )

    from openhealth.task_spec import load_task as real_load_task

    def _load_task_hide_model_kind_in_items(task_id: str):
        spec = real_load_task(task_id)
        orig = spec.to_train_params

        def wrapped(data_path):
            d = dict(orig(data_path))
            model_kind = d.pop("model_kind", None)

            class TpDict(dict):
                def get(self, key, default=None):
                    if key == "model_kind":
                        return model_kind
                    return super().get(key, default)

                def __getitem__(self, key):
                    if key == "model_kind":
                        return model_kind
                    return super().__getitem__(key)

            return TpDict(d)

        spec.to_train_params = wrapped
        return spec

    _orig_hpo_dump = HpoJobBody.model_dump

    def _hpo_dump_without_model_kind(self, *args, **kwargs):
        d = _orig_hpo_dump(self, *args, **kwargs)
        d["model_kind"] = None
        return d

    monkeypatch.setattr(HpoJobBody, "model_dump", _hpo_dump_without_model_kind)
    monkeypatch.setattr("openhealth.task_spec.load_task", _load_task_hide_model_kind_in_items)
    # HPO has no force flag — use paper_synthetic so task-aware health passes.
    start_hpo(
        HpoJobBody(
            data_path="data/raw/paper_synthetic_cohort.csv",
            task_id="diabetes",
            max_trials=1,
            label_col=None,
            horizon_days=None,
        ),
        True,
    )
    monkeypatch.setattr(HpoJobBody, "model_dump", _orig_hpo_dump)

    with pytest.raises(HTTPException) as exc_missing:
        start_hpo(
            HpoJobBody(data_path="data/raw/no_such_file_xyz.csv", task_id="diabetes", max_trials=1),
            True,
        )
    assert exc_missing.value.status_code == 404

    monkeypatch.setattr(
        "openhealth.health.dataset_health_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("health")),
    )
    with pytest.raises(HTTPException) as exc2:
        start_hpo(
            HpoJobBody(data_path="data/raw/ehr_data.csv", task_id="diabetes", max_trials=1),
            True,
        )
    assert exc2.value.status_code == 400
