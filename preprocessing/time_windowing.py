import pandas as pd


def filter_events_to_window(
    group: pd.DataFrame,
    index_time: pd.Timestamp,
    window_days: int,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Rows with timestamp in [index_time - window_days, index_time] (inclusive).

    For strict prediction-time semantics, pass a frame truncated to events
    strictly before the clinical index time; then index_time is last allowed event.
    """
    start = index_time - pd.Timedelta(days=window_days)
    t = group[time_col]
    return group[(t >= start) & (t <= index_time)]


def patient_index_times(df: pd.DataFrame, time_col: str = "timestamp") -> pd.Series:
    """Last observed time per patient (anchor for lookback window)."""
    return df.groupby("patient_id")[time_col].transform("max")

