"""Health gate + source tags."""

from openhealth.health import dataset_health_report


def test_health_ready_on_demo():
    report = dataset_health_report("data/raw/ehr_data.csv")
    h = report["health"]
    assert h["ready_for_training"] is True
    assert h.get("tiny_cohort") is True
    assert any("tiny_cohort" in w for w in h.get("warnings") or [])


def test_health_horizon_task_blocks_tiny_demo():
    report = dataset_health_report("data/demo/ehr_data.csv", task_id="horizon_detection_30d")
    h = report["health"]
    assert h["ready_for_training"] is False
    joined = " ".join(h.get("blockers") or []).lower()
    assert "index_time" in joined


def test_health_horizon_task_ok_on_paper_synthetic():
    report = dataset_health_report(
        "data/raw/paper_synthetic_cohort.csv", task_id="horizon_detection_30d"
    )
    h = report["health"]
    assert h["ready_for_training"] is True


def test_health_blocker_no_label(tmp_path):
    p = tmp_path / "nolabel.csv"
    p.write_text("patient_id,timestamp\n1,2023-01-01\n", encoding="utf-8")
    # save under project for relative resolve — write to uploads
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "nolabel_test.csv"
    dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    report = dataset_health_report(dest)
    assert report["health"]["ready_for_training"] is False
    assert report["health"]["blockers"]


def test_source_tags_api(client):
    r = client.get("/v1/datasets")
    assert r.status_code == 200
    ds = {d["id"]: d for d in r.json()["datasets"]}
    assert ds["paper_synthetic"].get("source_type") in ("synthetic", "demo")
    assert ds["ehr_data"].get("source_type") == "demo"
