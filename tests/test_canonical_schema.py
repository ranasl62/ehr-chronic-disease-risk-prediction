import pandas as pd

from preprocessing.canonical_schema import (
    assert_longitudinal_minimum,
    rename_to_canonical_longitudinal,
)


def test_rename_aliases():
    df = pd.DataFrame(
        {
            "BENE_ID": [1, 1],
            "CLAIM_DT": ["2020-01-01", "2020-02-01"],
            "dx_code": ["E11.9", "E11.9"],
            "glucose_mg_dl": [100.0, 105.0],
            "label": [0, 0],
        }
    )
    out = rename_to_canonical_longitudinal(df)
    assert "patient_id" in out.columns
    assert "timestamp" in out.columns
    assert "icd_code" in out.columns
    assert "glucose" in out.columns


def test_assert_minimum():
    df = rename_to_canonical_longitudinal(
        pd.DataFrame(
            {
                "member_id": [1],
                "service_date": ["2021-01-01"],
                "label": [0],
            }
        )
    )
    assert_longitudinal_minimum(df, need_label=True)
