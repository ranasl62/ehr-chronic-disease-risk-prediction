import math

import pytest

from inference.validation import build_input_stats_frame, validate_feature_dict


def test_validate_missing_key():
    with pytest.raises(ValueError, match="Missing"):
        validate_feature_dict({"a": 1.0}, required_columns=["a", "b"])


def test_validate_extra_key():
    with pytest.raises(ValueError, match="Unexpected"):
        validate_feature_dict({"a": 1.0, "b": 2.0}, required_columns=["a"])


def test_validate_nan():
    with pytest.raises(ValueError, match="finite"):
        validate_feature_dict({"a": float("nan")}, required_columns=["a"])


def test_validate_inf():
    with pytest.raises(ValueError, match="finite"):
        validate_feature_dict({"a": float("inf")}, required_columns=["a"])


def test_input_stats():
    import pandas as pd

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    st = build_input_stats_frame(X)
    assert "x" in st
    assert math.isfinite(st["x"]["median"])
