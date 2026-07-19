"""Exhaustive edge cases for schema mapping."""

from pathlib import Path

import pandas as pd
import pytest

from openhealth.schema_map import (
    apply_mapping,
    enrich_upload_with_aliases,
    map_import,
    map_preview_from_dataframe,
    map_preview_path,
    suggest_mapping,
)


def test_suggest_mapping_identity_canonical():
    cols = ["patient_id", "timestamp", "label", "glucose", "age"]
    sug = suggest_mapping(cols)
    assert sug["patient_id"] == "patient_id"
    assert sug["timestamp"] == "timestamp"
    assert sug["label"] == "label"
    assert sug["glucose"] == "glucose"


def test_suggest_mapping_unknown_columns_ignored():
    sug = suggest_mapping(["foo", "bar", "baz"])
    assert sug["patient_id"] is None
    assert sug["timestamp"] is None


def test_apply_mapping_renames_and_aliases():
    df = pd.DataFrame(
        [{"member_id": 1, "service_date": "2020-01-01", "outcome": 0, "blood_glucose": 100}]
    )
    mapped = apply_mapping(
        df,
        {
            "patient_id": "member_id",
            "timestamp": "service_date",
            "label": "outcome",
            "glucose": "blood_glucose",
        },
    )
    assert "patient_id" in mapped.columns
    assert "timestamp" in mapped.columns
    assert "label" in mapped.columns


def test_map_preview_reports_auto_alias_failure_without_patient():
    df = pd.DataFrame([{"x": 1, "y": 2}])
    prev = map_preview_from_dataframe(df)
    assert prev["auto_alias_ok"] is False
    assert prev["errors"]


def test_map_import_requires_patient_id(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame([{"a": 1}]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        map_import(p, {"patient_id": None, "timestamp": None, "label": None}, name="x.csv")


def test_map_import_missing_label_raises(tmp_path):
    p = tmp_path / "nolab.csv"
    pd.DataFrame(
        [
            {"member_id": 1, "service_date": "2020-01-01", "blood_glucose": 100},
            {"member_id": 2, "service_date": "2020-01-01", "blood_glucose": 120},
        ]
    ).to_csv(p, index=False)
    mapping = {
        "patient_id": "member_id",
        "timestamp": "service_date",
        "label": None,
        "chronic_disease": None,
        "glucose": "blood_glucose",
    }
    with pytest.raises(ValueError):
        map_import(p, mapping, name="no_label_map.csv")


def test_enrich_upload_aliases_person_id():
    df = pd.DataFrame(
        [{"person_id": 9, "charttime": "2021-01-01", "glucose_mg_dl": 120, "outcome": 1}]
    )
    out = enrich_upload_with_aliases(df)
    assert "patient_id" in out.columns
    assert "timestamp" in out.columns


def test_map_preview_path_missing_file():
    with pytest.raises(Exception):
        map_preview_path("data/raw/definitely_missing_xyz.csv")


def test_map_api_import_invalid_mapping(client):
    from utils.config import PROJECT_ROOT

    dest = PROJECT_ROOT / "data" / "uploads" / "map_edge.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"a": 1}]).to_csv(dest, index=False)
    r = client.post(
        "/v1/datasets/map-import",
        json={"path": "data/uploads/map_edge.csv", "mapping": {}, "name": "fail.csv"},
    )
    assert r.status_code == 400
