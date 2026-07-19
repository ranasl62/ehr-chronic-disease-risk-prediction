"""Exhaustive adapter and FHIR/OMOP failure paths."""

import json
from pathlib import Path

import pandas as pd
import pytest

from openhealth.adapters import (
    fhir_bundle_to_longitudinal,
    import_fhir_payload,
    import_omop_payload,
    load_fhir_file,
    omop_tables_to_longitudinal,
)


def test_omop_requires_person_id():
    with pytest.raises(ValueError, match="person_id"):
        omop_tables_to_longitudinal(pd.DataFrame([{"id": 1}]))


def test_omop_measurement_requires_date():
    person = pd.DataFrame([{"person_id": 1}])
    meas = pd.DataFrame([{"person_id": 1, "value_as_number": 1}])
    with pytest.raises(ValueError, match="measurement"):
        omop_tables_to_longitudinal(person, meas)


def test_omop_condition_only_path():
    person = pd.DataFrame([{"person_id": 1, "year_of_birth": 1990}])
    cond = pd.DataFrame(
        [{"person_id": 1, "condition_start_date": "2020-01-01", "condition_concept_id": 99}]
    )
    df = omop_tables_to_longitudinal(person, None, cond)
    assert len(df) == 1
    assert int(df.iloc[0]["label"]) == 1


def test_omop_empty_raises():
    person = pd.DataFrame([{"person_id": 1}])
    with pytest.raises(ValueError, match="no rows"):
        omop_tables_to_longitudinal(person, None, None)


def test_fhir_list_of_resources():
    resources = [
        {"resourceType": "Patient", "id": "a"},
        {
            "resourceType": "Observation",
            "subject": {"reference": "Patient/a"},
            "effectiveDateTime": "2022-01-01",
            "valueQuantity": {"value": 88},
        },
    ]
    df = fhir_bundle_to_longitudinal(resources)
    assert df.iloc[0]["patient_id"] == "a"


def test_fhir_single_resource_patient_only():
    df = fhir_bundle_to_longitudinal({"resourceType": "Patient", "id": "solo"})
    assert len(df) == 1
    assert df.iloc[0]["patient_id"] == "solo"


def test_fhir_invalid_payload():
    with pytest.raises(ValueError, match="Expected FHIR"):
        fhir_bundle_to_longitudinal({"foo": 1})


def test_fhir_empty_bundle_raises():
    with pytest.raises(ValueError, match="no Observation"):
        fhir_bundle_to_longitudinal({"resourceType": "Bundle", "entry": []})


def test_load_fhir_ndjson(tmp_path):
    p = tmp_path / "x.ndjson"
    p.write_text(
        json.dumps({"resourceType": "Patient", "id": "1"})
        + "\n"
        + json.dumps(
            {
                "resourceType": "Observation",
                "subject": {"reference": "Patient/1"},
                "effectiveDateTime": "2020-01-01",
                "valueQuantity": {"value": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    objs = load_fhir_file(p)
    assert isinstance(objs, list) and len(objs) == 2
    df = fhir_bundle_to_longitudinal(objs)
    assert len(df) >= 1


def test_import_omop_payload_persists(client):
    r = client.post(
        "/v1/datasets/from-omop",
        json={
            "person": [{"person_id": 3}],
            "measurement": [],
            "condition_occurrence": [
                {"person_id": 3, "condition_start_date": "2019-01-01", "condition_concept_id": 1}
            ],
            "name": "omop_cond_only.csv",
        },
    )
    assert r.status_code == 200


def test_import_omop_api_bad(client):
    r = client.post("/v1/datasets/from-omop", json={"person": [{"x": 1}], "name": "bad.csv"})
    assert r.status_code == 400


def test_import_fhir_api_bad(client):
    r = client.post("/v1/datasets/from-fhir", json={"bundle": {"foo": True}, "name": "bad.json"})
    assert r.status_code == 400
