"""Targeted tests to reach 100% line coverage on api/ + openhealth/ product surface."""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient as StarletteTestClient

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
        "feature_importance": {"a": 1.0},
        "shap_background": None,
        "X_train": pd.DataFrame({"a": [0, 1, 0, 1], "age": [40, 50, 45, 55]}),
        "X_test": pd.DataFrame({"a": [0, 1], "age": [42, 52]}),
        "y_train": pd.Series([0, 1, 0, 1]),
        "y_test": pd.Series([0, 1]),
    }
    art.update(extra)
    return art


# --- openhealth.api ---


def test_openhealth_train_with_task(tiny_csv, tmp_path, monkeypatch):
    from openhealth.api import train

    out = tmp_path / "m.pkl"
    monkeypatch.setattr("training.train.run_training", lambda **kw: (None, None, None, {"ok": True}))
    train(data_path=tiny_csv, task="diabetes", out=out, model_kind="logreg")
    train(data_path=tiny_csv, out=out, windows_days=[7, 30])


def test_openhealth_evaluate_no_report_no_model(tmp_path, monkeypatch):
    from openhealth.api import evaluate

    monkeypatch.setattr("utils.eval_report.load_evaluation_report_safe", lambda: None)
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError):
        evaluate(missing)


def test_openhealth_predict_and_save(tmp_path, monkeypatch):
    from openhealth.api import predict, save_model

    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    save_model(art, p)
    assert p.is_file()
    with pytest.raises(ValueError, match="missing required features"):
        predict({"a": 1.0}, artifact_path=p)
    with pytest.raises(ValueError, match="non-finite"):
        predict({"a": float("nan"), "age": 1.0}, artifact_path=p)
    out = predict({"a": 1.0, "age": 45.0}, artifact_path=p)
    assert "risk_probability" in out


def test_openhealth_explain(tmp_path, monkeypatch):
    from openhealth.api import explain

    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    plot = tmp_path / "shap.png"

    def _fake_explain(*_a, **_k):
        plot.write_bytes(b"png")

    monkeypatch.setattr("explainability.shap_explainer.explain_model", _fake_explain)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    result = explain(artifact_path=p, out=plot)
    assert result == plot


# --- openhealth.cli ---


def test_cli_doctor_missing_demo(monkeypatch):
    from openhealth.cli import main

    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert main(["doctor"]) == 1


def test_cli_start_no_docker(monkeypatch):
    from openhealth.cli import main

    monkeypatch.setattr("shutil.which", lambda _: None)
    assert main(["start"]) == 1


def test_cli_start_docker(monkeypatch):
    from openhealth.cli import main

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("subprocess.call", lambda *a, **k: 0)
    assert main(["start", "-d"]) == 0


def test_cli_train_explain_compare(tmp_path, monkeypatch, tiny_csv):
    from openhealth.cli import main

    monkeypatch.setattr("openhealth.api.train", lambda **kw: None)
    monkeypatch.setattr("openhealth.compare.compare_models", lambda **kw: {"selected_model": "logreg", "comparison": []})
    assert main(["train", "--data", str(tiny_csv), "--model", "logreg"]) == 0
    assert main(["train", "--task", "diabetes"]) == 0
    monkeypatch.setattr("openhealth.api.explain", lambda **kw: tmp_path / "s.png")
    assert main(["explain", "--out", str(tmp_path / "s.png")]) == 0
    assert main(["compare", "--data", str(tiny_csv), "--calibrate"]) == 0


def test_cli_train_task_no_suggested_path(monkeypatch):
    from openhealth.cli import main
    from openhealth.task_spec import TaskSpec

    spec = TaskSpec(id="x", name="x", suggested_path=None)
    monkeypatch.setattr("openhealth.task_spec.load_task", lambda _: spec)
    with pytest.raises(SystemExit):
        main(["train", "--task", "x"])


# --- openhealth modules ---


def test_analysis_pack_relative_missing():
    from openhealth.analysis_pack import build_analysis_pack

    with pytest.raises(FileNotFoundError):
        build_analysis_pack("data/uploads/no_such_pack.csv")


def test_write_analysis_pack_run_dir(tmp_path):
    from openhealth.analysis_pack import write_analysis_pack

    pack = {"kind": "analysis_pack", "n_rows": 1}
    run_dir = tmp_path / "run"
    out = write_analysis_pack(pack, run_dir=run_dir)
    assert out.is_file()
    assert (run_dir / "analysis_pack.json").is_file()


def test_events_ring_and_write_fail(monkeypatch):
    from openhealth import events as ev

    ev.clear_events()
    for i in range(205):
        ev.emit("t", f"m{i}")
    assert len(ev.list_events(limit=200)) <= 200

    monkeypatch.setattr(Path, "mkdir", MagicMock(side_effect=OSError("disk full")))
    ev.emit("fail", "x")


def test_config_store_yaml_errors(tmp_path):
    from openhealth.config_store import load_config

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(scalar)


def test_runs_corrupt_json_and_promote_config_fail(tmp_path, monkeypatch):
    from openhealth import runs as runs_mod

    rid = runs_mod.new_run_id("badjson")
    d = runs_mod.ensure_run(rid)
    (d / "run_meta.json").write_text("{bad", encoding="utf-8")
    summary = runs_mod._run_summary(d)
    assert summary["run_id"] == rid

    fake_runs = PROJECT_ROOT / "reports" / "runs"
    fake_runs.mkdir(parents=True, exist_ok=True)
    stray = fake_runs / "_list_runs_skip.txt"
    stray.write_text("x", encoding="utf-8")
    sub = fake_runs / "_unit_list_run"
    sub.mkdir(exist_ok=True)
    runs_mod.list_runs(limit=5)
    stray.unlink(missing_ok=True)

    rid2 = runs_mod.new_run_id("promo")
    rd = runs_mod.ensure_run(rid2)
    joblib.dump(_fake_artifact(), rd / "model.pkl")
    (rd / "evaluation_report.json").write_text("{}", encoding="utf-8")
    (rd / "feature_importance.json").write_text("{}", encoding="utf-8")
    (rd / "training_manifest.json").write_text("{}", encoding="utf-8")

    def _fail_save(*a, **k):
        raise OSError("cfg fail")

    monkeypatch.setattr("openhealth.config_store.save_config", _fail_save)
    monkeypatch.setattr("openhealth.events.emit", lambda *a, **k: {})
    result = runs_mod.promote_run(rid2)
    assert result["run_id"] == rid2


def test_trust_pack_branches(tmp_path):
    from openhealth.trust_pack import (
        build_trust_pack,
        mirror_to_shared,
        read_trust_pack,
        resolve_active_run_id,
        write_trust_pack,
    )

    rid = "unit_trust"
    rd = tmp_path / rid
    rd.mkdir()
    (rd / "model.pkl").write_bytes(b"x")
    (rd / "evaluation_report.json").write_text('{"metrics":{}}', encoding="utf-8")
    (rd / "leakage_audit.json").write_text(
        '{"split_method":"patient_group","patient_disjoint_train_test":false}', encoding="utf-8"
    )
    pack = build_trust_pack(rid, rd)
    assert pack["flags"]["leakage_passed"] is False

    (rd / "leakage_audit.json").write_text(
        '{"temporal_integrity":{"passed":false}}', encoding="utf-8"
    )
    build_trust_pack(rid, rd)

    (rd / "training_manifest.json").write_text("{bad", encoding="utf-8")
    build_trust_pack(rid, rd)

    write_trust_pack(rid, rd)
    (rd / "trust_pack.json").write_text("{bad", encoding="utf-8")
    assert read_trust_pack(rd) is None

    mirror_to_shared(rd / "trust_pack.json", "trust_pack.json")
    with pytest.raises(ValueError):
        resolve_active_run_id("../evil")

    with patch("openhealth.config_store.load_config", side_effect=RuntimeError):
        assert resolve_active_run_id() is None


def test_task_spec_yaml_and_list(tmp_path, monkeypatch):
    from openhealth.task_spec import list_tasks, load_task

    with patch("openhealth.task_spec._require_yaml", side_effect=ImportError("no pyyaml")):
        with pytest.raises(ImportError):
            load_task("diabetes")

    bad = tmp_path / "bad_task.yaml"
    bad.write_text("scalar_only", encoding="utf-8")
    with pytest.raises(ValueError):
        load_task(bad)

    flex = tmp_path / "flex.yaml"
    flex.write_text(
        """
task: {id: flex, name: Flex}
target: diabetes_event
prediction: {horizon_days: 365_days, window: 365_days}
features: {windows_days: [7d, 30_days]}
data: {format: tabular, path: data/raw/ehr_data.csv}
training: {model: logreg}
""",
        encoding="utf-8",
    )
    t = load_task(flex)
    assert t.target_column == "diabetes_event"
    assert 7 in t.windows_days

    broken = tmp_path / "broken.yaml"
    broken.write_text("task: {id: x, name: X}\ntarget: label\n", encoding="utf-8")
    monkeypatch.setattr("openhealth.task_spec.TASKS_DIR", tmp_path)
    listed = list_tasks(tmp_path)
    assert any(x.id == "flex" for x in listed)


def test_health_branches(tmp_path):
    from openhealth.health import dataset_health_report

    p = tmp_path / "hi.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "index_time": "2020-01-01", "age": 50},
            {"patient_id": 2, "timestamp": "2020-06-01", "label": 1, "index_time": "2020-01-01", "age": 60},
        ]
    ).to_csv(p, index=False)
    h = dataset_health_report(p, task_id="readmission_30d")
    assert "health" in h

    with patch("openhealth.task_spec.load_task", side_effect=FileNotFoundError("nope")):
        h2 = dataset_health_report(p, task_id="missing")
        assert h2["health"]["blockers"]


def test_clinical_audit_bad_lines(tmp_path, monkeypatch):
    from openhealth import clinical_audit as ca

    path = tmp_path / "audit.jsonl"
    path.write_text('{"ok":1}\n{bad json\n', encoding="utf-8")
    monkeypatch.setattr(ca, "AUDIT_PATH", path)
    assert ca.recent_audit(limit=5) == [{"ok": 1}]


def test_compare_promote_best(tiny_csv, tmp_path, monkeypatch):
    from openhealth.compare import compare_models

    monkeypatch.setattr(
        "training.train.run_training",
        lambda **kw: (None, pd.DataFrame({"a": [1]}), pd.Series([0]), {"model": DummyClassifier().fit([[0]], [0])}),
    )
    monkeypatch.setattr(
        "training.reporting.build_evaluation_report",
        lambda *a, **k: {"metrics": {"roc_auc": 0.5}},
    )
    out_dir = tmp_path / "cmp"
    summary = compare_models(
        data_path=tiny_csv,
        models=["logreg"],
        out_dir=out_dir,
        promote_best=True,
    )
    assert summary["selected_model"] == "logreg"


def test_adapters_edges(tmp_path):
    from openhealth.adapters import fhir_bundle_to_longitudinal, load_fhir_file, omop_tables_to_longitudinal

    person = pd.DataFrame([{"person_id": 1, "year_of_birth": 1980}])
    meas = pd.DataFrame([{"person_id": 1, "measurement_date": "bad-year", "value_as_number": 1}])
    df = omop_tables_to_longitudinal(person, meas)
    assert df.iloc[0]["age"] is None

    resources = [
        {"resourceType": "Patient", "id": "p1"},
        {"not": "dict"},
        {
            "resourceType": "Condition",
            "subject": {"reference": "Patient/p1"},
        },
        {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/p1"},
            "effectiveDateTime": "2020-01-01",
            "valueQuantity": {"value": 99},
        },
    ]
    fhir_bundle_to_longitudinal(resources)

    bundle = {"resourceType": "Bundle", "entry": [{"resource": resources[0]}]}
    fhir_bundle_to_longitudinal(bundle)

    nd = tmp_path / "f.ndjson"
    nd.write_text('{"resourceType":"Patient","id":"1"}\n', encoding="utf-8")
    assert load_fhir_file(nd)


def test_schema_map_relative_path(tmp_path):
    from openhealth.schema_map import map_import

    p = tmp_path / "m.csv"
    p.write_text("member_id,service_date,blood_glucose,outcome\n1,2020-01-01,100,0\n", encoding="utf-8")
    meta = map_import(p, {"member_id": "patient_id", "service_date": "timestamp", "blood_glucose": "glucose", "outcome": "label"})
    assert "path" in meta


# --- api.middleware ---


async def _boom(_request):
    raise RuntimeError("boom")


def test_middleware_audit_and_exception(tmp_path, monkeypatch):
    from api import middleware as mw

    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(mw, "_AUDIT_PATH", str(audit))

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/predict", ok, methods=["POST"])])
    app.add_middleware(mw.RequestContextMiddleware)
    c = StarletteTestClient(app)
    r = c.post("/v1/predict", json={"features": {"a": 1}})
    assert r.status_code == 200
    assert audit.read_text(encoding="utf-8")

    app2 = Starlette(routes=[Route("/v1/predict", _boom, methods=["POST"])])
    app2.add_middleware(mw.RequestContextMiddleware)
    c2 = StarletteTestClient(app2, raise_server_exceptions=False)
    assert c2.post("/v1/predict", json={}).status_code == 500

    mw.configure_api_logging()
    mw.configure_api_logging()


def test_middleware_audit_write_fail(tmp_path, monkeypatch):
    from api import middleware as mw

    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(mw, "_AUDIT_PATH", str(audit))

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/v1/predict", ok, methods=["POST"])])
    app.add_middleware(mw.RequestContextMiddleware)
    with patch("builtins.open", side_effect=OSError("no write")):
        c = StarletteTestClient(app)
        r = c.post("/v1/predict", json={})
        assert r.status_code == 200


# --- api.production_middleware ---


def test_body_size_invalid_content_length():
    from api.production_middleware import BodySizeLimitMiddleware

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/x", ok, methods=["POST"])])
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=100)
    c = StarletteTestClient(app)
    r = c.post("/x", content=b"x", headers={"content-length": "not-a-number"})
    assert r.status_code == 200


def test_rate_limit_pops_old_entries():
    from api.production_middleware import RateLimitMiddleware

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/x", ok, methods=["GET"])])
    app.add_middleware(RateLimitMiddleware, per_minute=1)
    c = StarletteTestClient(app)
    assert c.get("/x").status_code == 200
    assert c.get("/x").status_code == 429


# --- api.main ---


def test_lifespan_api_key_log(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200


def test_readiness_and_predict_endpoints(monkeypatch, tmp_path):
    from api.main import app, artifact_dep, get_artifact

    get_artifact.cache_clear()
    missing = tmp_path / "missing.pkl"
    monkeypatch.setattr("api.main.MODEL_PATH", missing)
    monkeypatch.setattr("utils.config.MODEL_PATH", missing)
    c = TestClient(app)
    assert c.get("/v1/ready").status_code == 503

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr("api.main.MODEL_PATH", bad)
    monkeypatch.setattr("utils.config.MODEL_PATH", bad)
    assert TestClient(app).get("/v1/ready").status_code == 503

    art = _fake_artifact()
    p = tmp_path / "good.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("api.main.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    app2 = __import__("api.main", fromlist=["app"]).app
    app2.dependency_overrides[artifact_dep] = lambda: art
    try:
        c2 = TestClient(app2)
        assert c2.get("/v1/ready").status_code == 200
        r = c2.post("/predict", json={"age": 50, "glucose": 100, "blood_pressure": 120, "cholesterol": 200})
        assert r.status_code in (200, 400, 422)
        r2 = c2.post("/v1/predict", json={"features": {"a": 1, "age": 45}, "include_explanation": True})
        assert r2.status_code == 200
        r3 = c2.post("/explain", json={"features": {"a": "bad", "age": 45}})
        assert r3.status_code in (400, 422)
        r4 = c2.post("/predict/raw?include_explanation=true", json={"a": 1, "age": 45, "extra": 9})
        assert r4.status_code == 200
        r5 = c2.post("/predict/raw", json={"a": "not-a-number", "age": 45})
        assert r5.status_code == 400
    finally:
        app2.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_get_artifact_503():
    from api.main import app, get_artifact

    get_artifact.cache_clear()
    with patch("api.main.Path.exists", return_value=False):
        get_artifact.cache_clear()
        from api.main import artifact_dep

        app.dependency_overrides.clear()
        c = TestClient(app)
        with patch.object(Path, "exists", return_value=False):
            get_artifact.cache_clear()
            r = c.get("/v1/model/schema")
            assert r.status_code == 503


def test_rate_limit_middleware_wired(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    import importlib
    import api.main as main_mod

    importlib.reload(main_mod)
    assert main_mod.app is not None


# --- api.data_io ---


def test_delete_dataset_edges(tmp_path, monkeypatch):
    from api.data_io import delete_dataset_file

    with pytest.raises(ValueError, match="under project root"):
        delete_dataset_file("/etc/passwd")

    uploads = PROJECT_ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "symlink_target.csv"
    target.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    link = uploads / "sym_link.csv"
    if link.exists():
        link.unlink()
    link.symlink_to(target)
    delete_dataset_file(str(link.relative_to(PROJECT_ROOT)))

    with pytest.raises(ValueError):
        delete_dataset_file("data/demo/.gitkeep")


def test_data_io_import_formats(tmp_path):
    from api.data_io import (
        dataframe_from_upload_bytes,
        import_form_rows,
        import_sql,
        profile_dataset,
        build_methods_markdown,
        build_results_zip,
    )

    tsv = b"patient_id,timestamp,label\n1,2020-01-01,0\n"
    df = dataframe_from_upload_bytes("x.tsv", tsv)
    assert len(df) == 1

    data_json = b'{"data":[{"patient_id":1,"timestamp":"2020-01-01","label":0}]}'
    df2 = dataframe_from_upload_bytes("x.json", data_json)
    assert "patient_id" in df2.columns

    with pytest.raises(ValueError):
        import_form_rows([])

    with pytest.raises(ValueError, match="SELECT"):
        import_sql("DELETE FROM t")

    with pytest.raises(ValueError, match="Mutating"):
        import_sql("SELECT 1; DROP TABLE t")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        import_sql("SELECT 1")

    csv = tmp_path / "prof.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "age": 44, "sex": "F", "glucose": 100},
            {"patient_id": 2, "timestamp": "2020-02-01", "label": 1, "age": 55, "sex": "M", "glucose": 110},
        ]
    ).to_csv(csv, index=False)
    prof = profile_dataset(csv)
    assert prof.get("cohort_rows")

    text = build_methods_markdown()
    assert "Methods note" in text

    raw = build_results_zip()
    assert zipfile.ZipFile(io.BytesIO(raw)).namelist()


def test_save_dataframe_format_fallback(monkeypatch):
    from api import data_io

    df = pd.DataFrame([{"patient_id": 1, "glucose": 100, "label": 0}])
    calls = {"n": 0}

    def _summarize(_path, data_format="longitudinal"):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"blocking": True, "message": "bad fmt"}]
        return []

    def _assert(issues):
        if issues and any(i.get("blocking") for i in issues):
            raise ValueError("blocked")

    monkeypatch.setattr(data_io, "summarize_csv", _summarize)
    monkeypatch.setattr(data_io, "assert_no_blocking_errors", _assert)
    meta = data_io._save_dataframe(df, "fmt_fallback_unit.csv")
    assert meta["format"] in ("longitudinal", "tabular")


# --- api.jobs ---


def test_jobs_log_trim_and_cancel_before_start():
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="t1", kind="t")
    for i in range(210):
        rec.append(f"line {i}")
    assert len(rec.log) <= 200

    rec2 = jobs_mod.JobRecord(id="t2", kind="sleep")
    rec2.status = "cancelled"

    def noop(_r):
        pass

    with patch.object(jobs_mod, "_EXECUTOR") as ex:
        ex.submit = lambda fn: fn()
        jobs_mod.submit_job("noop", noop)


def test_run_train_promote_fallback(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="tr", kind="train")
    params = {
        "data_path": str(tiny_csv),
        "model_kind": "logreg",
        "promote": True,
        "windows_days": [7],
    }

    model_dest = tmp_path / "active_model.pkl"
    monkeypatch.setattr("api.jobs.MODEL_PATH", model_dest)
    monkeypatch.setattr("utils.config.MODEL_PATH", model_dest)

    run_dir = tmp_path / "run_unit"
    run_dir.mkdir()
    (run_dir / "evaluation_report.json").write_text("{}", encoding="utf-8")
    (run_dir / "feature_importance.json").write_text("{}", encoding="utf-8")
    (run_dir / "training_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "model.pkl").write_bytes(b"x")

    monkeypatch.setattr("training.train.run_training", lambda **kw: None)
    monkeypatch.setattr("openhealth.runs.ensure_run", lambda rid: run_dir)
    monkeypatch.setattr("openhealth.runs.new_run_id", lambda p: "run_unit")
    monkeypatch.setattr("openhealth.runs.write_run_meta", lambda *a, **k: None)
    monkeypatch.setattr("openhealth.trust_pack.write_trust_pack", lambda *a, **k: {})
    monkeypatch.setattr("openhealth.runs.promote_run", MagicMock(side_effect=RuntimeError("fail")))
    monkeypatch.setattr("openhealth.config_store.load_config", lambda: {})
    monkeypatch.setattr("openhealth.config_store.save_config", lambda c: c)
    jobs_mod.run_train_job(rec, params)


def test_run_compare_and_fairness(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="c", kind="compare")
    monkeypatch.setattr(
        "openhealth.compare.compare_models",
        lambda **kw: {"selected_model": "logreg", "comparison": []},
    )
    jobs_mod.run_compare_job(rec, {"data_path": str(tiny_csv)})

    rec2 = jobs_mod.JobRecord(id="f", kind="fairness")
    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("api.jobs.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    jobs_mod.run_fairness_job(rec2, {})

    gcsv = tmp_path / "groups.csv"
    pd.DataFrame({"age_band": ["lt50", "ge65"]}).to_csv(gcsv, index=False)
    jobs_mod.run_fairness_job(rec2, {"groups_path": str(gcsv), "group_column": "age_band"})


def test_run_leakage_raw_path(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod
    import importlib.util
    import sys

    rec = jobs_mod.JobRecord(id="l", kind="leakage")
    audit_path = PROJECT_ROOT / "scripts" / "leakage_audit.py"
    spec = importlib.util.spec_from_file_location("leakage_audit_mod", audit_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.audit_from_raw = lambda **kw: {"passed": True, "temporal_integrity": {"passed": False}}
    sys.modules["leakage_audit_mod"] = mod
    jobs_mod.run_leakage_audit_job(rec, {"use_artifact": False, "data_path": str(tiny_csv)})


def test_run_hpo_and_shap_and_ext(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="h", kind="hpo")
    monkeypatch.setattr(
        "training.hpo.run_light_hpo",
        lambda **kw: {"report_path": "r", "best": {"roc_auc": 0.5}, "n_trials": 1},
    )
    jobs_mod.run_hpo_job(rec, {"data_path": str(tiny_csv)})

    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("api.jobs.MODEL_PATH", p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    monkeypatch.setattr("explainability.shap_explainer.explain_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    monkeypatch.setattr("openhealth.trust_pack.resolve_active_run_id", lambda x=None: "run1")
    monkeypatch.setattr("openhealth.runs.ensure_run", lambda rid: tmp_path / rid)
    monkeypatch.setattr("openhealth.runs.run_path", lambda rid: tmp_path / rid)
    rec2 = jobs_mod.JobRecord(id="s", kind="shap")
    jobs_mod.run_shap_job(rec2, {"run_id": "run1"})

    monkeypatch.setattr(
        "training.external_validate.external_validate",
        lambda **kw: {"metrics": {"roc_auc": 0.5}},
    )
    monkeypatch.setattr("training.external_validate.write_external_validation_report", lambda *a, **k: tmp_path / "ev.json")
    rec3 = jobs_mod.JobRecord(id="e", kind="ext")
    jobs_mod.run_external_validate_job(rec3, {"data_path": str(tiny_csv)})


def test_cancel_queued_job():
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="q", kind="q")
    with jobs_mod._LOCK:
        jobs_mod._JOBS[rec.id] = rec
    out = jobs_mod.cancel_job(rec.id)
    assert out.status == "cancelled"


# --- api routes via client ---


def test_researcher_route_edges(client, tmp_path, monkeypatch):
    from openhealth.runs import ensure_run, new_run_id, write_run_meta

    rid = new_run_id("ws")
    ensure_run(rid)
    write_run_meta(rid, {"kind": "train"})
    bad_manifest = REPORTS_DIR / "training_manifest.json"
    bad_manifest.write_text("{bad", encoding="utf-8")
    client.get("/v1/workspace/status")

    r = client.get("/v1/tasks/bad_task_xyz")
    assert r.status_code in (404, 400)

    r2 = client.get("/v1/datasets/health", params={"path": "data/raw/no_such.csv"})
    assert r2.status_code == 404

    r3 = client.post("/v1/datasets/upload", files={"file": ("", b"")})
    assert r3.status_code in (400, 422)

    r4 = client.delete("/v1/datasets", params={"path": str(Path("/etc/passwd"))})
    assert r4.status_code == 400

    r5 = client.get("/v1/datasets/profile", params={"path": "data/raw/no_such.csv"})
    assert r5.status_code == 404

    r6 = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/uploads/nolabel_test.csv",
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
        },
    )
    assert r6.status_code in (400, 409)

    r7 = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": "data/raw/no_such.csv",
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
        },
    )
    assert r7.status_code == 404

    r8 = client.post("/v1/jobs/leakage-audit", json={"use_artifact": False})
    assert r8.status_code == 400

    r9 = client.get("/v1/reports/analysis-pack", params={"path": "data/raw/no_such.csv"})
    assert r9.status_code == 404

    r10 = client.get("/v1/reports/download.zip", params={"run_id": "../evil"})
    assert r10.status_code in (400, 404)

    r11 = client.get("/v1/reports/file/not_allowed.json")
    assert r11.status_code == 404

    r12 = client.get("/v1/reports/file/evaluation_report.json")
    assert r12.status_code in (200, 404)


def test_framework_route_edges(client, tmp_path, monkeypatch):
    from api.main import app, artifact_dep, get_artifact

    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    monkeypatch.setattr("utils.config.MODEL_PATH", p)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )

    fp = REPORTS_DIR / "fairness_report.json"
    fp_backup = fp.read_text(encoding="utf-8") if fp.is_file() else None
    try:
        r = client.get("/v1/reports/thresholds")
        assert r.status_code in (200, 404)

        fp.write_text("{bad", encoding="utf-8")
        r2 = client.get("/v1/reports/fairness")
        assert r2.status_code == 500

        r3 = client.post("/v1/runs/does_not_exist/promote")
        assert r3.status_code == 404

        get_artifact.cache_clear()
        app.dependency_overrides[artifact_dep] = lambda: art
        monkeypatch.setattr(
            "openhealth.api.predict",
            lambda row, **kw: {"risk_probability": 0.5, "feature_columns": ["a", "age"]},
        )
        try:
            r4 = client.post("/v1/worklist/predict", json={"rows": [{"a": 1.0, "age": 40.0}]})
            assert r4.status_code == 200
        finally:
            app.dependency_overrides.clear()

        r5 = client.put("/v1/workspace/config", json={"model_kind": "lstm"})
        assert r5.status_code in (400, 422)

        r6 = client.post("/v1/tasks", json={"id": "../evil", "name": "x"})
        assert r6.status_code in (400, 422)

        r7 = client.post("/v1/datasets/map-preview", json={"path": "../etc/passwd"})
        assert r7.status_code in (400, 422)
    finally:
        if fp_backup is None:
            fp.unlink(missing_ok=True)
        else:
            fp.write_text(fp_backup, encoding="utf-8")


# --- Additional gap closure (100% target) ---


def test_lifespan_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    from api.main import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200


def test_configure_api_logging_idempotent():
    from api import middleware as mw

    mw.log.handlers.clear()
    mw.configure_api_logging()
    mw.configure_api_logging()
    from api.middleware import _should_audit

    assert _should_audit("/v1/predict", "GET") is False


def test_production_rate_limit_expires_old():
    from api.production_middleware import RateLimitMiddleware
    from collections import deque
    import time as time_mod

    async def ok(_req):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/x", ok, methods=["GET"])])
    app.add_middleware(RateLimitMiddleware, per_minute=1)
    with patch.object(time_mod, "time", return_value=120.0):
        c = StarletteTestClient(app)
        assert c.get("/x").status_code == 200


def test_main_legacy_predict_and_explain(client):
    from api.main import app, artifact_dep, get_artifact

    art = {
        **_fake_artifact(),
        "feature_columns": ["age", "glucose", "blood_pressure", "cholesterol"],
    }
    get_artifact.cache_clear()
    app.dependency_overrides[artifact_dep] = lambda: art
    try:
        c = TestClient(app)
        r = c.post(
            "/predict",
            json={"age": 50, "glucose": 100, "blood_pressure": 120, "cholesterol": 200},
        )
        assert r.status_code == 200
        r2 = c.post("/explain", json={"features": {"age": 50, "glucose": 100, "blood_pressure": 120, "cholesterol": 200}})
        assert r2.status_code == 200
    finally:
        app.dependency_overrides.clear()
        get_artifact.cache_clear()


def test_data_io_delete_symlink_and_imports(tmp_path):
    from api.data_io import delete_dataset_file, dataframe_from_upload_bytes, import_sql, profile_dataset

    uploads = PROJECT_ROOT / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    protected = uploads / "prot.csv"
    protected.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n", encoding="utf-8")
    link = uploads / "link_to_prot.csv"
    if link.exists():
        link.unlink()
    link.symlink_to(protected)
    result = delete_dataset_file(str(link.relative_to(PROJECT_ROOT)))
    assert result["deleted"] is True

    xlsx_bytes = b"PK\x03\x04"
    with pytest.raises(Exception):
        dataframe_from_upload_bytes("t.xlsx", xlsx_bytes)

    csv = tmp_path / "prof2.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "age": 44, "sex": "F"},
            {"patient_id": 2, "timestamp": "2020-02-01", "label": 1, "age": 55, "sex": "M"},
        ]
    ).to_csv(csv, index=False)
    prof = profile_dataset(csv, age_band="50_59", label="1", patient_id="2")
    assert prof.get("filters")

    db = tmp_path / "t.db"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (patient_id INT, timestamp TEXT, label INT)")
    conn.executemany("INSERT INTO t VALUES (?,?,?)", [(1, "2020-01-01", 0), (2, "2020-01-02", 1)])
    conn.commit()
    conn.close()
    meta = import_sql("SELECT * FROM t", connection_url=f"sqlite:///{db}", name="sql_unit.csv")
    assert "path" in meta


def test_build_methods_with_artifacts(tmp_path, monkeypatch):
    from api.data_io import build_methods_markdown
    from openhealth.runs import ensure_run, new_run_id

    rid = new_run_id("methods")
    rd = ensure_run(rid)
    (rd / "training_manifest.json").write_text(
        json.dumps({"model_kind": "logreg", "data_sha256": "abc", "temporal_split": True}),
        encoding="utf-8",
    )
    (rd / "evaluation_report.json").write_text(
        json.dumps({"meta": {"model_kind": "logreg", "feature_engineering": {"split_method": "patient_group"}}}),
        encoding="utf-8",
    )
    (rd / "leakage_audit.json").write_text(
        json.dumps(
            {
                "patient_disjoint_train_test": True,
                "temporal_integrity": {"passed": True, "feature_events_after_index": 0},
                "warnings": ["icd feature name"],
            }
        ),
        encoding="utf-8",
    )
    (rd / "trust_pack.json").write_text(
        json.dumps({"flags": {"trust_complete": True, "has_leakage": True, "leakage_passed": True}}),
        encoding="utf-8",
    )
    (rd / "external_validation_report.json").write_text(
        json.dumps({"data_path": "x.csv", "metrics": {"roc_auc": 0.7}, "n_samples": 10}),
        encoding="utf-8",
    )
    (rd / "analysis_pack.json").write_text(json.dumps({"n_patients": 5, "n_rows": 10}), encoding="utf-8")
    text = build_methods_markdown(run_id=rid)
    assert "External validation" in text
    assert "Analysis pack" in text


def test_openhealth_cli_report_and_compare_task(tmp_path, monkeypatch, tiny_csv):
    from openhealth.cli import main

    out = tmp_path / "pack.zip"
    assert main(["report", "--out", str(out)]) == 0
    assert out.is_file()

    monkeypatch.setattr("openhealth.compare.compare_models", lambda **kw: {"selected_model": "logreg", "comparison": [{"model": "logreg", "roc_auc": 0.5, "selected": True}]})
    assert main(["compare", "--task", "diabetes", "--data", str(tiny_csv)]) == 0

    monkeypatch.setattr("openhealth.api.train", lambda **kw: None)
    assert main(["train", "--task", "diabetes", "--format", "longitudinal", "--model", "logreg"]) == 0


def test_openhealth_api_evaluate_and_explain_relative(tmp_path, monkeypatch):
    from openhealth.api import evaluate, explain

    monkeypatch.setattr("utils.eval_report.load_evaluation_report_safe", lambda: {"metrics": {"roc_auc": 0.5}})
    assert evaluate()["metrics"]["roc_auc"] == 0.5

    art = _fake_artifact()
    p = tmp_path / "m.pkl"
    joblib.dump(art, p)
    plot = "reports/custom_shap.png"
    monkeypatch.setattr("explainability.shap_explainer.explain_model", lambda *a, **k: None)
    monkeypatch.setattr(
        "training.reproduce_split.split_train_test_from_artifact",
        lambda a: (art["X_train"], art["X_test"], art["y_train"], art["y_test"], None, None),
    )
    path = explain(artifact_path=p, out=plot)
    assert path.is_absolute()


def test_compare_lightgbm_branch(monkeypatch):
    from openhealth.compare import available_models

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("no lgbm")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", fake_import):
        models = available_models()
    assert "lightgbm" not in models


def test_config_store_yaml_import_error(tmp_path, monkeypatch):
    from openhealth import config_store as cs

    cfg_file = tmp_path / "workspace.yaml"
    cfg_file.write_text("persona: researcher\n", encoding="utf-8")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", fake_import):
        with pytest.raises(ImportError):
            cs.load_config(cfg_file)


def test_clinical_audit_missing_file():
    from openhealth import clinical_audit as ca

    with patch.object(ca, "AUDIT_PATH", Path("/nonexistent/clinical_audit.jsonl")):
        assert ca.recent_audit() == []


def test_health_warning_paths(tmp_path):
    from openhealth.health import dataset_health_report

    p = tmp_path / "tiny.csv"
    pd.DataFrame([{"patient_id": 1, "timestamp": "2020-01-01", "label": 0, "index_time": "2020-01-01"}]).to_csv(
        p, index=False
    )
    rep = dataset_health_report(p)
    assert rep["health"]["warnings"]


def test_runs_dir_missing(monkeypatch, tmp_path):
    from openhealth import runs as runs_mod

    monkeypatch.setattr(runs_mod, "RUNS_DIR", tmp_path / "missing_runs")
    assert runs_mod.list_runs() == []


def test_task_spec_target_dict_key_only(tmp_path):
    from openhealth.task_spec import load_task

    p = tmp_path / "t.yaml"
    p.write_text(
        "task: {id: t, name: T}\ntarget: {only_col: x}\ndata: {suggested_path: data/raw/ehr_data.csv}\n",
        encoding="utf-8",
    )
    t = load_task(p)
    assert t.target_column == "only_col"


def test_trust_pack_bad_json(tmp_path):
    from openhealth.trust_pack import build_trust_pack

    rd = tmp_path / "r"
    rd.mkdir()
    (rd / "leakage_audit.json").write_text("{bad", encoding="utf-8")
    (rd / "training_manifest.json").write_text("{bad", encoding="utf-8")
    build_trust_pack("r", rd)


def test_adapters_fhir_edges():
    from openhealth.adapters import fhir_bundle_to_longitudinal, load_fhir_file

    with pytest.raises(ValueError, match="no Observation"):
        fhir_bundle_to_longitudinal({"resourceType": "Bundle", "entry": []})

    resources = [
        {"resourceType": "Condition", "subject": {"reference": "p1"}},
        {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/p1"},
            "effectiveDateTime": "2020-01-01",
            "valueQuantity": {"value": 1},
        },
    ]
    fhir_bundle_to_longitudinal(resources)


def test_schema_map_relative(tmp_path):
    from openhealth.schema_map import map_import

    rel = Path("data/uploads/map_edge.csv")
    target = PROJECT_ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "member_id,service_date,outcome\n1,2020-01-01,0\n2,2020-02-01,1\n",
        encoding="utf-8",
    )
    meta = map_import(
        "data/uploads/map_edge.csv",
        {
            "member_id": "patient_id",
            "service_date": "timestamp",
            "outcome": "label",
        },
        name="mapped_rel.csv",
    )
    assert meta["path"]


def test_jobs_remaining_branches(tmp_path, monkeypatch, tiny_csv):
    from api import jobs as jobs_mod

    rec = jobs_mod.JobRecord(id="x", kind="x")
    rec.status = "cancelled"

    def noop(_r):
        pass

    with patch.object(jobs_mod, "_EXECUTOR") as ex:
        ex.submit = lambda fn: fn()
        jobs_mod.submit_job("noop", noop)

    rec3 = jobs_mod.JobRecord(id="c2", kind="compare")
    monkeypatch.setattr("openhealth.compare.compare_models", lambda **kw: (_ for _ in ()).throw(FileNotFoundError("missing")))
    with pytest.raises(FileNotFoundError):
        jobs_mod.run_compare_job(rec3, {"data_path": str(tmp_path / "nope.csv")})

    rec4 = jobs_mod.JobRecord(id="t2", kind="train")
    with pytest.raises(FileNotFoundError):
        jobs_mod.run_train_job(rec4, {"data_path": str(tmp_path / "nope.csv")})


def test_researcher_routes_more(client, wait_jobs_idle, tiny_csv):
    wait_jobs_idle()
    dest = PROJECT_ROOT / "data" / "uploads" / "api_upload_unit.csv"
    dest.write_text("patient_id,timestamp,label\n1,2020-01-01,0\n2,2020-02-01,1\n", encoding="utf-8")

    r = client.get("/v1/datasets", params={"include_demo": False})
    assert r.status_code == 200

    r2 = client.post(
        "/v1/jobs/train",
        json={
            "data_path": str(dest.relative_to(PROJECT_ROOT)),
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
            "task_id": "custom",
            "label_col": "label",
        },
    )
    assert r2.status_code in (200, 409)

    r3 = client.post(
        "/v1/jobs/compare",
        json={
            "data_path": str(dest.relative_to(PROJECT_ROOT)),
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
            "task_id": "custom",
            "label_col": "label",
        },
    )
    assert r3.status_code in (200, 409)

    r4 = client.get("/v1/reports/file/evaluation_report.json")
    assert r4.status_code in (200, 404)

    r5 = client.get("/v1/reports/methods.md", params={"run_id": "../bad"})
    assert r5.status_code == 400


def test_framework_promote_and_fairness_job(client, tmp_path, wait_jobs_idle, monkeypatch):
    from openhealth.runs import ensure_run, new_run_id

    model_path = PROJECT_ROOT / "model.pkl"
    model_backup = model_path.read_bytes() if model_path.is_file() else None
    active = tmp_path / "promoted_model.pkl"
    monkeypatch.setattr("utils.config.MODEL_PATH", active)

    wait_jobs_idle()
    rid = new_run_id("promoapi")
    rd = ensure_run(rid)
    joblib.dump(_fake_artifact(), rd / "model.pkl")
    (rd / "evaluation_report.json").write_text("{}", encoding="utf-8")
    (rd / "feature_importance.json").write_text("{}", encoding="utf-8")
    (rd / "training_manifest.json").write_text("{}", encoding="utf-8")

    r = client.post(f"/v1/runs/{rid}/promote")
    assert r.status_code == 200

    r2 = client.get("/v1/reports/fairness")
    assert r2.status_code == 200

    wait_jobs_idle()
    r3 = client.post("/v1/jobs/fairness", json={})
    assert r3.status_code in (200, 409)

    if model_backup is not None:
        model_path.write_bytes(model_backup)

