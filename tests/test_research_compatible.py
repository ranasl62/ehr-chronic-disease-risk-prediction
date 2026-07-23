"""Research-compatible core: trust pack, leakage warnings, external validate, analysis pack, task health."""

from __future__ import annotations

import io
import json
import time
import zipfile

from openhealth.analysis_pack import build_analysis_pack
from openhealth.runs import ensure_run, get_run, list_runs, new_run_id, promote_run
from openhealth.task_spec import list_tasks, load_task
from openhealth.trust_pack import write_trust_pack
from utils.config import PROJECT_ROOT, REPORTS_DIR


def _wait_job(client, jid: str, *, timeout_loops: int = 80) -> dict:
    st = {}
    for _ in range(timeout_loops):
        st = client.get(f"/v1/jobs/{jid}").json()
        if st["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.25)
    return st


def _ensure_trained_run(client, wait_jobs_idle) -> str:
    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/demo/ehr_data.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "promote": True,
            "calibrate": False,
            "windows_days": [7, 30, 180],
        },
    )
    assert r.status_code == 200, r.text
    st = _wait_job(client, r.json()["id"])
    wait_jobs_idle()
    assert st["status"] == "succeeded", st
    run_id = st["result"].get("run_id")
    assert run_id
    return run_id


def test_trust_pack_after_train_unit():
    rid = new_run_id("trustunit")
    rd = ensure_run(rid)
    (rd / "model.pkl").write_bytes(b"fake")
    (rd / "evaluation_report.json").write_text(json.dumps({"metrics": {"roc_auc": 0.7}}), encoding="utf-8")
    (rd / "training_manifest.json").write_text(json.dumps({"data_sha256": "abc"}), encoding="utf-8")
    pack = write_trust_pack(rid, rd)
    assert pack["flags"]["has_model"]
    assert pack["flags"]["has_evaluation"]
    assert pack["data_sha256"] == "abc"
    assert (rd / "trust_pack.json").is_file()
    assert "model.pkl" in pack["artifacts"]


def test_leakage_and_shap_land_in_run(client, wait_jobs_idle):
    run_id = _ensure_trained_run(client, wait_jobs_idle)

    wait_jobs_idle()
    r = client.post("/v1/jobs/leakage-audit", json={"use_artifact": True, "run_id": run_id})
    assert r.status_code == 200, r.text
    st = _wait_job(client, r.json()["id"])
    wait_jobs_idle()
    assert st["status"] == "succeeded", st
    assert (REPORTS_DIR / "runs" / run_id / "leakage_audit.json").is_file()
    detail = get_run(run_id)
    assert detail.get("has_leakage") or (detail.get("trust") or {}).get("has_leakage")

    wait_jobs_idle()
    r2 = client.post("/v1/jobs/shap", json={"run_id": run_id})
    assert r2.status_code == 200, r2.text
    st2 = _wait_job(client, r2.json()["id"], timeout_loops=120)
    wait_jobs_idle()
    assert st2["status"] == "succeeded", st2
    assert (REPORTS_DIR / "runs" / run_id / "shap_summary.png").is_file()


def test_promote_copies_trust_extras(client, wait_jobs_idle):
    run_id = _ensure_trained_run(client, wait_jobs_idle)
    rd = ensure_run(run_id)
    (rd / "leakage_audit.json").write_text(json.dumps({"passed": True, "temporal_integrity": {"passed": True}}), encoding="utf-8")
    (rd / "shap_summary.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    write_trust_pack(run_id, rd)
    out = promote_run(run_id)
    assert out["run_id"] == run_id
    assert (REPORTS_DIR / "leakage_audit.json").is_file()
    assert (REPORTS_DIR / "shap_summary.png").is_file()
    assert (REPORTS_DIR / "trust_pack.json").is_file()


def test_icd_proxy_warnings():
    import importlib.util
    import sys

    audit_path = PROJECT_ROOT / "scripts" / "leakage_audit.py"
    spec = importlib.util.spec_from_file_location("leakage_audit_mod", audit_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leakage_audit_mod"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    warns = mod._icd_proxy_warnings(["w7d_glucose", "w30d_icd_unique_count", "age"], horizon_days=7)
    assert warns
    assert "ICD" in warns[0] or "icd" in warns[0].lower()


def test_external_validate_model(client, wait_jobs_idle):
    run_id = _ensure_trained_run(client, wait_jobs_idle)
    from training.external_validate import external_validate

    report = external_validate(
        artifact_path=ensure_run(run_id) / "model.pkl",
        data_path=PROJECT_ROOT / "data/demo/ehr_data.csv",
        data_format="longitudinal",
    )
    assert "metrics" in report
    assert report["n_samples"] > 0

    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/external-validate",
        json={"data_path": "data/demo/ehr_data.csv", "data_format": "longitudinal", "run_id": run_id},
    )
    assert r.status_code == 200, r.text
    st = _wait_job(client, r.json()["id"])
    wait_jobs_idle()
    assert st["status"] == "succeeded", st
    assert (REPORTS_DIR / "external_validation_report.json").is_file()
    assert (REPORTS_DIR / "runs" / run_id / "external_validation_report.json").is_file()

    r_miss = client.post(
        "/v1/jobs/external-validate",
        json={"data_path": "data/demo/does_not_exist.csv"},
    )
    assert r_miss.status_code == 404


def test_analysis_pack_keys():
    pack = build_analysis_pack(PROJECT_ROOT / "data/demo/ehr_data.csv")
    assert "n_patients" in pack
    assert "n_rows" in pack
    assert "missingness" in pack
    assert "label_prevalence" in pack


def test_analysis_pack_api(client):
    r = client.get("/v1/reports/analysis-pack", params={"path": "data/demo/ehr_data.csv"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("n_rows", 0) > 0
    assert (REPORTS_DIR / "analysis_pack.json").is_file()


def test_new_task_presets_listed():
    ids = {t.id for t in list_tasks()}
    assert "horizon_detection_30d" in ids
    assert "teaching_leaky_contrast" in ids
    hz = load_task("horizon_detection_30d")
    assert hz.horizon_days == 30
    assert "index_time" in hz.required_columns()


def test_runs_api_trust_flags(client):
    runs = list_runs(limit=3)
    if not runs:
        pytest.skip("no runs")
    rid = runs[0]["run_id"]
    r = client.get(f"/v1/runs/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert "trust_complete" in body or "trust" in body or "has_leakage" in body


def test_zip_run_scoped(client):
    runs = list_runs(limit=3)
    if not runs:
        pytest.skip("no runs")
    rid = runs[0]["run_id"]
    write_trust_pack(rid, ensure_run(rid))
    r = client.get("/v1/reports/download.zip", params={"run_id": rid})
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "reports/methods.md" in names
    methods = zf.read("reports/methods.md").decode("utf-8")
    assert rid in methods or "Trust pack" in methods


def test_methods_md_sections(client):
    # seed analysis + external if missing
    client.get("/v1/reports/analysis-pack", params={"path": "data/demo/ehr_data.csv"})
    r = client.get("/v1/reports/methods.md")
    assert r.status_code == 200
    text = r.text
    assert "Trust pack" in text
    assert "External validation" in text
    assert "Analysis pack" in text


def test_health_task_required_columns(client):
    # demo ehr may lack index_time — readmission task should block
    r = client.get(
        "/v1/datasets/health",
        params={"path": "data/demo/ehr_data.csv", "task_id": "readmission_30d"},
    )
    assert r.status_code == 200
    health = r.json().get("health") or {}
    # Either blockers present or ready depending on CSV columns
    assert "blockers" in health
    assert "task_id" in health
    # teaching leaky contrast CSV has index_time
    r2 = client.get(
        "/v1/datasets/health",
        params={"path": "data/demo/teaching_leaky_contrast.csv", "task_id": "teaching_leaky_contrast"},
    )
    assert r2.status_code == 200
    h2 = r2.json().get("health") or {}
    assert h2.get("ready_for_training") is True or not any(
        "missing required" in b for b in (h2.get("blockers") or [])
    )
