from preprocessing.data_quality import check_longitudinal, check_tabular, summarize_csv


def test_longitudinal_ok():
    import pandas as pd

    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="30D"),
            "label": [0, 0, 1, 1],
        }
    )
    issues = check_longitudinal(df)
    assert not any(i.get("level") == "error" for i in issues)


def test_longitudinal_missing_label():
    import pandas as pd

    df = pd.DataFrame({"patient_id": [1], "timestamp": ["2020-01-01"]})
    issues = check_longitudinal(df)
    assert any(i.get("code") == "missing_label" for i in issues)


def test_summarize_csv_longitudinal(tmp_path):
    import pandas as pd

    p = tmp_path / "t.csv"
    pd.DataFrame(
        {
            "patient_id": [1, 1],
            "timestamp": ["2020-01-01", "2020-02-01"],
            "label": [0, 0],
        }
    ).to_csv(p, index=False)
    issues = summarize_csv(p, data_format="longitudinal")
    assert not any(i.get("level") == "error" for i in issues)


def test_tabular_missing():
    import pandas as pd

    issues = check_tabular(pd.DataFrame({"patient_id": [1]}))
    assert any("chronic_disease" in i.get("message", "") for i in issues)
