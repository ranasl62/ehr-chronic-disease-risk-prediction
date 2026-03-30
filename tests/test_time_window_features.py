import pandas as pd

from feature_engineering.time_window_features import create_time_window_features


def test_time_window_aggregates_last_180_days():
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2020-12-01"]
            ),
            "glucose": [100.0, 110.0, 120.0],
            "blood_pressure": [120.0, 122.0, 124.0],
            "cholesterol": [180.0, 185.0, 190.0],
            "age": [50, 50, 51],
            "icd_code": ["A", "B", "B"],
        }
    )
    out = create_time_window_features(df, window_days=180)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["patient_id"] == 1
    assert row["visit_count"] == 1.0
    assert row["glucose"] == 120.0
    assert row["icd_unique_count"] == 1.0
