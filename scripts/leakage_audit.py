#!/usr/bin/env python3
"""
Leakage and split sanity checks for longitudinal / tabular training data.

Does not replace a full cohort audit; encodes minimum checks before MIMIC-scale runs.

  PYTHONPATH=. python scripts/leakage_audit.py --format longitudinal --data data/raw/ehr_data.csv --split-by-patient
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from preprocessing.data_quality import assert_no_blocking_errors, summarize_csv
from preprocessing.cleaning import clean_longitudinal_ehr
from preprocessing.ehr_loader import load_data, load_ehr_data
from training.reproduce_split import load_xy_groups_from_artifact, reproducible_train_test_indices
from training.splits import temporal_patient_train_test_indices
from training.train import build_xy_longitudinal, build_xy_tabular


def _patient_disjoint(train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray) -> bool:
    g_tr = set(groups[train_idx])
    g_te = set(groups[test_idx])
    return g_tr.isdisjoint(g_te)


def audit_from_raw(
    *,
    data_path: Path,
    data_format: str,
    split_by_patient: bool,
    temporal_split: bool,
    test_size: float,
    random_state: int,
    window_days: int,
    windows: str | None,
) -> dict:
    issues = summarize_csv(data_path, data_format=data_format)
    assert_no_blocking_errors(issues)

    if data_format == "longitudinal":
        df = load_ehr_data(data_path)
        if windows:
            wd = tuple(int(x.strip()) for x in windows.split(",") if x.strip())
            X, y, _, groups = build_xy_longitudinal(df, windows_days=wd)
        else:
            X, y, _, groups = build_xy_longitudinal(df, window_days=window_days)
    else:
        df = load_data(data_path)
        X, y, _, groups = build_xy_tabular(df)

    if temporal_split:
        if data_format != "longitudinal":
            raise ValueError("--temporal-split requires longitudinal format.")
        df_c = clean_longitudinal_ehr(load_ehr_data(data_path))
        last_e = df_c.groupby("patient_id")["timestamp"].max()
        tr_i, te_i = temporal_patient_train_test_indices(
            np.asarray(groups), last_e, test_size=test_size
        )
        split_method = "temporal_patient"
    else:
        split_method = "patient_group" if split_by_patient else "random_row"
        tr_i, te_i = reproducible_train_test_indices(
            X,
            y,
            np.asarray(groups),
            split_method=split_method,
            test_size=test_size,
            random_state=random_state,
            fe=None,
        )
    disjoint = _patient_disjoint(tr_i, te_i, np.asarray(groups))
    out = {
        "data_path": str(data_path.resolve()),
        "format": data_format,
        "split_method": split_method,
        "n_rows": int(len(X)),
        "n_patients": int(len(np.unique(groups))),
        "train_rows": int(len(tr_i)),
        "test_rows": int(len(te_i)),
        "patient_disjoint_train_test": disjoint,
        "data_quality_issues": issues,
        "notes": [],
    }
    if data_format == "longitudinal" and not split_by_patient and not temporal_split:
        out["notes"].append(
            "Row-level split on longitudinal features risks same-patient leakage; prefer --split-by-patient or --temporal-split."
        )
    if temporal_split:
        out["notes"].append(
            "Temporal split: test patients have the latest activity times — closer to external validation than random group split."
        )
    if split_by_patient and not disjoint:
        out["notes"].append("CRITICAL: patient overlap between train and test despite patient_group split.")
    if data_format == "longitudinal":
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            last_per_patient = df.assign(_ts=ts).groupby(df["patient_id"].values)["_ts"].max()
            out["timestamp_parse_ok"] = True
            out["patients_with_valid_time"] = int(last_per_patient.notna().sum())
    return out


def audit_from_artifact(artifact_path: Path) -> dict:
    import joblib

    art = joblib.load(artifact_path)
    fe = art.get("feature_engineering") or {}
    X, y, groups = load_xy_groups_from_artifact(art)
    split_method = fe.get("split_method", "random_row")
    test_size = float(fe.get("test_size", 0.2))
    random_state = int(fe.get("random_state", 42))
    tr_i, te_i = reproducible_train_test_indices(
        X,
        y,
        groups,
        split_method=split_method,
        test_size=test_size,
        random_state=random_state,
        fe=fe,
    )
    disjoint = _patient_disjoint(tr_i, te_i, groups)
    notes: list[str] = []
    if split_method in ("patient_group", "temporal_patient") and not disjoint:
        notes.append("Patient overlap in train/test — investigate split logic.")
    if split_method == "random_row" and fe.get("format") == "longitudinal":
        notes.append("Row-level split on longitudinal features may leak patients across train/test.")
    return {
        "artifact": str(artifact_path.resolve()),
        "split_method": split_method,
        "n_rows": int(len(X)),
        "n_patients": int(len(np.unique(groups))),
        "patient_disjoint_train_test": disjoint,
        "notes": notes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Leakage / split sanity audit.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path, help="Training CSV (with --format)")
    g.add_argument("--artifact", type=Path, help="model.pkl from a completed train run")
    p.add_argument("--format", choices=["longitudinal", "tabular"], default="longitudinal")
    sp = p.add_mutually_exclusive_group()
    sp.add_argument("--split-by-patient", action="store_true")
    sp.add_argument("--temporal-split", action="store_true")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-days", type=int, default=180)
    p.add_argument("--windows", type=str, default=None, help="e.g. 7,30,180")
    p.add_argument("-o", "--out-json", type=Path, default=None)
    args = p.parse_args()

    if args.artifact:
        report = audit_from_artifact(args.artifact)
    else:
        report = audit_from_raw(
            data_path=args.data,
            data_format=args.format,
            split_by_patient=args.split_by_patient,
            temporal_split=args.temporal_split,
            test_size=args.test_size,
            random_state=args.seed,
            window_days=args.window_days,
            windows=args.windows,
        )

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out_json:
        args.out_json.write_text(text, encoding="utf-8")

    sm = report.get("split_method")
    if sm in ("patient_group", "temporal_patient") and not report.get("patient_disjoint_train_test", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
