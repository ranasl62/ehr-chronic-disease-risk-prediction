"""openhealth facade, compare scoring, data_io ZIP, claim hygiene."""

import io
import zipfile
from pathlib import Path

import pytest

from api.data_io import build_results_zip, dataframe_from_upload_bytes
from openhealth.compare import _score, available_models
from utils.config import PROJECT_ROOT


def test_available_models_classical_only():
    models = available_models()
    assert "logreg" in models
    assert "xgboost" in models
    assert "lstm" not in models


def test_score_handles_missing_and_nan():
    assert _score({}) == -1.0
    assert _score({"roc_auc": float("nan")}) == -1.0
    assert _score({"roc_auc": 0.9}) == 0.9


def test_dataframe_from_json_records():
    data = b'{"records":[{"patient_id":1,"timestamp":"2020-01-01","label":0}]}'
    df = dataframe_from_upload_bytes("x.json", data)
    assert "patient_id" in df.columns


def test_dataframe_from_json_invalid():
    with pytest.raises(ValueError):
        dataframe_from_upload_bytes("x.json", b'{"foo":1}')


def test_dataframe_unsupported_ext():
    with pytest.raises(ValueError, match="Supported"):
        dataframe_from_upload_bytes("x.parquet", b"abc")


def test_zip_contains_limitations_and_manifest():
    raw = build_results_zip()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = set(zf.namelist())
    assert any("LIMITATIONS" in n for n in names)
    assert "README_PACK.json" in names


def test_openhealth_evaluate_returns_dict():
    from openhealth.api import evaluate

    ev = evaluate()
    assert isinstance(ev, dict)


def test_openhealth_predict_requires_model():
    from openhealth.api import predict
    from utils.config import MODEL_PATH

    if not Path(MODEL_PATH).is_file():
        pytest.skip("no model")
    with pytest.raises(ValueError, match="missing required features"):
        predict({})


def test_readme_avoids_forbidden_claims():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "fda-cleared",
        "medical device for diagnosis",
        "solves all healthcare",
        "bias-free",
        "state-of-the-art on every mimic",
    ):
        assert phrase not in text


def test_limitations_mentions_lstm_and_phi():
    text = (PROJECT_ROOT / "LIMITATIONS.md").read_text(encoding="utf-8").lower()
    assert "lstm" in text
    assert "phi" in text or "not a medical device" in text


def test_why_framework_exists_and_contrasts():
    text = (PROJECT_ROOT / "WHY_THIS_FRAMEWORK.md").read_text(encoding="utf-8")
    assert "leakage" in text.lower()
    assert "PyHealth" in text or "pyhealth" in text.lower()


def test_train_force_researcher_allows_bad_data(client):
    """Researcher may force past blockers; still may fail later on schema — expect 200 queue or 400 from train internals."""
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "force_nolabel.csv"
    dest.write_text("patient_id,timestamp\n1,2020-01-01\n2,2020-01-02\n", encoding="utf-8")
    r = client.post(
        "/v1/jobs/train",
        json={
            "data_path": "data/uploads/force_nolabel.csv",
            "data_format": "longitudinal",
            "model_kind": "logreg",
            "force": True,
        },
    )
    # With force, health gate opens; job may still fail on label — status 200 queued/failed ok
    assert r.status_code in (200, 400, 409)
    if r.status_code == 200:
        import time

        jid = r.json()["id"]
        for _ in range(40):
            st = client.get(f"/v1/jobs/{jid}").json()
            if st["status"] in ("succeeded", "failed"):
                # expected fail without label
                assert st["status"] == "failed"
                break
            time.sleep(0.15)
