import json
import math

from utils.json_safe import json_safe


def test_json_safe_replaces_non_finite_floats():
    out = json_safe({"roc_auc": float("nan"), "brier": 0.2, "nested": [1.0, float("inf")]})
    assert json.dumps(out) == '{"roc_auc": null, "brier": 0.2, "nested": [1.0, null]}'
    assert math.isnan(float("nan"))  # sanity
    assert out["roc_auc"] is None
