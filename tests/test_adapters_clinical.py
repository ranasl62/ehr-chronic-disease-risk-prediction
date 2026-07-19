"""Adapters + clinical worklist + claim hygiene."""

from openhealth.adapters import fhir_bundle_to_longitudinal, omop_tables_to_longitudinal
import pandas as pd


def test_omop_adapter_fixture():
    person = pd.DataFrame([{"person_id": 1, "year_of_birth": 1980}, {"person_id": 2, "year_of_birth": 1970}])
    measurement = pd.DataFrame(
        [
            {"person_id": 1, "measurement_date": "2020-01-01", "value_as_number": 100},
            {"person_id": 2, "measurement_date": "2020-01-01", "value_as_number": 140},
        ]
    )
    condition = pd.DataFrame([{"person_id": 2, "condition_start_date": "2020-02-01", "condition_concept_id": 1}])
    df = omop_tables_to_longitudinal(person, measurement, condition)
    assert "patient_id" in df.columns and "timestamp" in df.columns


def test_fhir_adapter_fixture():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {
                "resource": {
                    "resourceType": "Observation",
                    "subject": {"reference": "Patient/p1"},
                    "effectiveDateTime": "2020-01-01",
                    "valueQuantity": {"value": 110},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "subject": {"reference": "Patient/p1"},
                }
            },
        ],
    }
    df = fhir_bundle_to_longitudinal(bundle)
    assert len(df) >= 1
    assert df.iloc[0]["patient_id"] == "p1"


def test_adapter_api(client):
    r = client.post(
        "/v1/datasets/from-omop",
        json={
            "person": [{"person_id": 1, "year_of_birth": 1980}],
            "measurement": [
                {"person_id": 1, "measurement_date": "2020-01-01", "value_as_number": 100}
            ],
            "name": "omop_api_test.csv",
        },
    )
    assert r.status_code == 200
    assert r.json().get("source_type") == "omop"

    r2 = client.post(
        "/v1/datasets/from-fhir",
        json={
            "bundle": {
                "resourceType": "Bundle",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Observation",
                            "subject": {"reference": "Patient/x"},
                            "effectiveDateTime": "2021-01-01",
                            "valueQuantity": {"value": 90},
                        }
                    }
                ],
            },
            "name": "fhir_api_test.csv",
        },
    )
    assert r2.status_code == 200


def test_worklist_requires_disclaimer(client):
    from openhealth.config_store import load_config, save_config

    cfg = load_config()
    cfg["persona"] = "clinical_research"
    cfg["disclaimer_ack"] = False
    save_config(cfg)
    r = client.post("/v1/worklist/predict", json={"rows": [{"w7d_age": 50.0}]})
    assert r.status_code == 403
    cfg["persona"] = "researcher"
    cfg["disclaimer_ack"] = False
    save_config(cfg)


def test_worklist_empty(client):
    r = client.post("/v1/worklist/predict", json={"rows": []})
    assert r.status_code == 400


def test_no_lstm_in_framework_meta(client):
    r = client.get("/v1/meta/framework")
    assert r.status_code == 200
    body = r.json()
    assert "lstm" not in body["supported_models"]
    assert "lstm" in body["unsupported_models"]


def test_cli_doctor():
    from openhealth.cli import main

    assert main(["doctor"]) == 0


def test_why_and_limitations_exist():
    from utils.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "LIMITATIONS.md").is_file()
    assert (PROJECT_ROOT / "WHY_THIS_FRAMEWORK.md").is_file()
