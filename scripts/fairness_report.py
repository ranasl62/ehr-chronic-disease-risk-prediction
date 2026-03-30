#!/usr/bin/env python3
"""
Fairness-style metrics on the training holdout (same split as in model.pkl).

Provide a CSV mapping patient_id -> subgroup label (e.g. age band). Example:
  patient_id,age_band
  100,young
  200,old

  PYTHONPATH=. python scripts/fairness_report.py --artifact model.pkl --subgroups path/to/groups.csv --group-column age_band
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd

from fairness.bias_metrics import (
    binary_rates_by_group,
    demographic_parity_difference,
    subgroup_metrics_table,
)
from training.reproduce_split import load_xy_groups_from_artifact, split_train_test_from_artifact
from utils.config import MODEL_PATH, REPORTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=MODEL_PATH)
    ap.add_argument("--subgroups", type=Path, required=True, help="CSV with patient_id + group column")
    ap.add_argument("--group-column", type=str, required=True)
    ap.add_argument("--patient-column", type=str, default="patient_id")
    ap.add_argument("-o", type=Path, default=REPORTS_DIR / "fairness_holdout.json")
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    model = art["model"]
    _, X_test, _, y_test, _, te_idx = split_train_test_from_artifact(art)
    _, _, groups = load_xy_groups_from_artifact(art)
    groups = np.asarray(groups)
    test_patients = groups[te_idx]

    sub = pd.read_csv(args.subgroups)
    if args.patient_column not in sub.columns or args.group_column not in sub.columns:
        raise SystemExit(f"CSV must contain {args.patient_column!r} and {args.group_column!r}")
    sub = sub.copy()
    sub["_pk"] = sub[args.patient_column].astype(str)
    pmap = sub.drop_duplicates("_pk").set_index("_pk")[args.group_column].to_dict()
    row_group = np.array([pmap.get(str(pid), np.nan) for pid in test_patients])
    mask = pd.notna(row_group)
    if mask.sum() < 2:
        raise SystemExit("Too few test rows with subgroup labels after join.")

    mask_bool = np.asarray(mask, dtype=bool)
    X_sub = X_test.iloc[mask_bool]
    y_t = y_test.iloc[mask_bool].to_numpy()
    g_sub = row_group[mask_bool]
    prob = model.predict_proba(X_sub)[:, 1]
    pred = (prob >= 0.5).astype(int)

    dpd = demographic_parity_difference(pred, g_sub)
    table = subgroup_metrics_table(y_t, pred, prob, g_sub)
    rates = binary_rates_by_group(y_t, pred, g_sub)
    out = {
        "demographic_parity_difference": float(dpd) if dpd == dpd else None,
        "subgroup_table": table.to_dict(orient="records"),
        "tpr_fpr_by_group": rates.to_dict(orient="records"),
        "n_test_scored": int(len(y_t)),
        "artifact": str(args.artifact.resolve()),
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
