"""Schema map unit + API tests."""

from pathlib import Path

import pandas as pd

from openhealth.schema_map import map_import, map_preview_from_dataframe, map_preview_path, suggest_mapping


def test_suggest_mapping_aliases(messy_csv):
    df = pd.read_csv(messy_csv)
    sug = suggest_mapping(list(df.columns))
    assert sug["patient_id"] == "member_id"
    assert sug["timestamp"] == "service_date"
    assert sug["label"] == "outcome" or sug["glucose"] == "blood_glucose"


def test_map_preview_auto_alias(messy_csv):
    df = pd.read_csv(messy_csv)
    prev = map_preview_from_dataframe(df)
    assert "suggested_mapping" in prev
    assert prev["n_rows"] == 3


def test_map_import_writes_upload(messy_csv):
    prev = map_preview_path(messy_csv)
    mapping = prev["suggested_mapping"]
    # ensure label mapped
    if not mapping.get("label") and not mapping.get("chronic_disease"):
        mapping["label"] = "outcome"
    meta = map_import(messy_csv, mapping, name="test_mapped.csv")
    assert Path(meta["path"]).name == "test_mapped.csv"
    assert meta.get("source_type") == "byo"


def test_map_api(client, messy_csv, tmp_path, monkeypatch):
    # copy messy into project uploads-accessible path under data/uploads
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "messy_test.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(messy_csv.read_bytes())
    r = client.post("/v1/datasets/map-preview", json={"path": "data/uploads/messy_test.csv"})
    assert r.status_code == 200
    mapping = r.json()["suggested_mapping"]
    mapping["label"] = mapping.get("label") or "outcome"
    r2 = client.post(
        "/v1/datasets/map-import",
        json={
            "path": "data/uploads/messy_test.csv",
            "mapping": mapping,
            "name": "mapped_api.csv",
        },
    )
    assert r2.status_code == 200
    assert "path" in r2.json()


def test_map_api_bad_body(client):
    r = client.post("/v1/datasets/map-preview", json={"path": "data/raw/nope.csv"})
    assert r.status_code == 400
