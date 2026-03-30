import pandas as pd


def aggregate_numeric(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    how: str = "mean",
) -> pd.DataFrame:
    return df.groupby(group_col, as_index=False)[value_cols].agg(how)
