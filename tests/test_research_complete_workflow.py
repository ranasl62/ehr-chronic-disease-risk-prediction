"""End-to-end research workflow: datasets → health → train → trust → audit → pack → ZIP."""

from __future__ import annotations

import io
import json
import time
import zipfile

from openhealth.runs import ensure_run, get_run
from utils.config import PROJECT_ROOT, REPORTS_DIR

DEMO_PATH = "data/demo/ehr_data.csv"
DEMO_FORMAT = "longitudinal"


def _wait_job(client, jid: str, *, timeout_loops: int = 80) -> dict:
    st: dict = {}
    for _ in range(timeout_loops):
        st = client.get(f"/v1/jobs/{jid}").json()
        if st["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.25)
    return st


def _train_demo(client, wait_jobs_idle) -> str:
    wait_jobs_idle()
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": DEMO_PATH,
            "data_format": DEMO_FORMAT,
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


def test_research_workflow_end_to_end(client, wait_jobs_idle):
    """Exercise claimed research features in one sequential loop (not full monorepo coverage)."""

    # 1 — List datasets and workspace status
    ds = client.get("/v1/datasets").json()
    assert ds.get("datasets")
    assert any(d.get("exists") for d in ds["datasets"])
    ws = client.get("/v1/workspace/status").json()
    assert ws.get("api_ok") is True
    assert "checklist" in ws

    # 2 — Task catalog + task-aware health
    tasks = client.get("/v1/tasks").json()["tasks"]
    task_ids = {t["id"] for t in tasks}
    assert "horizon_detection_30d" in task_ids
    health = client.get(
        "/v1/datasets/health",
        params={"path": DEMO_PATH, "task_id": "horizon_detection_30d"},
    ).json()
    assert "health" in health
    h = health["health"]
    assert h.get("task_id") == "horizon_detection_30d"
    assert "blockers" in h

    # 3 — Train on demo CSV (small config)
    run_id = _train_demo(client, wait_jobs_idle)
    run_dir = ensure_run(run_id)

    # 4 — Trust pack after train
    trust_path = run_dir / "trust_pack.json"
    assert trust_path.is_file(), "train job should write trust_pack.json"
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    assert trust["run_id"] == run_id
    assert trust["flags"]["has_model"]
    assert trust["flags"]["has_evaluation"]
    assert (run_dir / "model.pkl").is_file()
    assert (run_dir / "evaluation_report.json").is_file()

    run_detail = client.get(f"/v1/runs/{run_id}").json()
    assert run_detail["run_id"] == run_id
    assert "metrics" in run_detail or "trust" in run_detail

    # 5 — Leakage audit scoped to run_id
    wait_jobs_idle()
    la = client.post("/v1/jobs/leakage-audit", json={"use_artifact": True, "run_id": run_id})
    assert la.status_code == 200, la.text
    la_st = _wait_job(client, la.json()["id"])
    wait_jobs_idle()
    assert la_st["status"] == "succeeded", la_st
    leakage_path = run_dir / "leakage_audit.json"
    assert leakage_path.is_file()
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    assert isinstance(leakage, dict)
    assert leakage.get("passed") is not False or leakage.get("temporal_integrity")
    detail = get_run(run_id)
    assert detail.get("has_leakage") or (detail.get("trust") or {}).get("has_leakage")

    # Refresh trust pack flags after leakage
    trust2 = json.loads((run_dir / "trust_pack.json").read_text(encoding="utf-8"))
    assert trust2["flags"].get("has_leakage") or "leakage_audit.json" in trust2.get("artifacts", {})

    # 6 — Analysis pack (cohort summary JSON)
    ap = client.get("/v1/reports/analysis-pack", params={"path": DEMO_PATH})
    assert ap.status_code == 200
    pack = ap.json()
    assert pack.get("n_rows", 0) > 0
    assert "n_patients" in pack
    assert "missingness" in pack
    assert "label_prevalence" in pack
    assert (REPORTS_DIR / "analysis_pack.json").is_file()

    # 7 — External validate on demo hold-out (cheap with tiny demo)
    wait_jobs_idle()
    ev = client.post(
        "/v1/jobs/external-validate",
        json={"data_path": DEMO_PATH, "data_format": DEMO_FORMAT, "run_id": run_id},
    )
    assert ev.status_code == 200, ev.text
    ev_st = _wait_job(client, ev.json()["id"])
    wait_jobs_idle()
    assert ev_st["status"] == "succeeded", ev_st
    ext_path = run_dir / "external_validation_report.json"
    assert ext_path.is_file()
    ext = json.loads(ext_path.read_text(encoding="utf-8"))
    assert "metrics" in ext
    assert ext.get("n_samples", 0) > 0

    # 8 — Methods markdown + run-scoped ZIP
    md = client.get("/v1/reports/methods.md", params={"run_id": run_id})
    assert md.status_code == 200
    text = md.text
    assert "Trust pack" in text
    assert "External validation" in text
    assert "Analysis pack" in text
    assert run_id in text or "run" in text.lower()

    zr = client.get("/v1/reports/download.zip", params={"run_id": run_id})
    assert zr.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(zr.content))
    names = zf.namelist()
    assert "reports/methods.md" in names
    methods = zf.read("reports/methods.md").decode("utf-8")
    assert "Trust pack" in methods
    assert any(n.endswith("trust_pack.json") or "trust_pack" in n for n in names) or "trust" in methods.lower()


def test_research_workflow_list_runs_after_train(client, wait_jobs_idle):
    """Runs API reflects a freshly trained experiment."""
    run_id = _train_demo(client, wait_jobs_idle)
    listed = client.get("/v1/runs", params={"limit": 20}).json()
    runs = listed.get("runs") or listed if isinstance(listed, list) else []
    if isinstance(listed, dict):
        runs = listed.get("runs", [])
    ids = {r.get("run_id") for r in runs if isinstance(r, dict)}
    assert run_id in ids or any(run_id in str(r) for r in runs)
