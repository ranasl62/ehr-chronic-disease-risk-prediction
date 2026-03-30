import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "age": "mean",
        "glucose": "mean",
        "blood_pressure": "mean",
        "cholesterol": "mean",
    }
    cols = {k: v for k, v in agg_map.items() if k in df.columns}
    if not cols:
        raise ValueError("Expected numeric EHR columns for aggregation.")
    features = df.groupby("patient_id", as_index=False).agg(cols)
    return features
