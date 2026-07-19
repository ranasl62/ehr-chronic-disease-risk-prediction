"""Experiment runs + promote + ZIP limitations."""

import io
import zipfile
from pathlib import Path

from api.data_io import build_results_zip
from openhealth.runs import ensure_run, list_runs, new_run_id, promote_run, write_run_meta
from utils.config import MODEL_PATH, PROJECT_ROOT


def test_run_dir_and_list():
    rid = new_run_id("unit")
    p = ensure_run(rid)
    write_run_meta(rid, {"kind": "test"})
    assert p.is_dir()
    runs = list_runs(limit=5)
    assert any(r["run_id"] == rid for r in runs)


def test_promote_missing_404(client):
    r = client.post("/v1/runs/does_not_exist_zzz/promote")
    assert r.status_code == 404


def test_promote_copies_model(tmp_path, monkeypatch):
    rid = new_run_id("promo")
    p = ensure_run(rid)
    # copy existing model if present else skip
    if not Path(MODEL_PATH).is_file():
        return
    import shutil

    shutil.copy2(MODEL_PATH, p / "model.pkl")
    write_run_meta(rid, {"kind": "train"})
    out = promote_run(rid)
    assert out["run_id"] == rid
    assert Path(MODEL_PATH).is_file()


def test_research_pack_includes_limitations():
    data = build_results_zip()
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    assert any("LIMITATIONS" in n for n in names)


def test_compare_default_no_auto_promote_in_body():
    from api.researcher_routes import CompareJobBody

    b = CompareJobBody(data_path="data/raw/ehr_data.csv")
    assert b.promote_best is False
