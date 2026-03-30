#!/usr/bin/env python3
"""
Patient-level GroupKFold CV — mean / std hold-out ROC-AUC (and Brier when defined).

  PYTHONPATH=. python scripts/group_cv_report.py --format longitudinal --data data/raw/ehr_data.csv --model logreg
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from preprocessing.ehr_loader import load_data, load_ehr_data
from training.train import build_xy_longitudinal, build_xy_tabular, fit_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--format", choices=["longitudinal", "tabular"], default="longitudinal")
    ap.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--windows", type=str, default="7,30,180")
    ap.add_argument("-o", type=Path, default=_ROOT / "reports" / "cv_group_metrics.json")
    args = ap.parse_args()

    if args.format == "longitudinal":
        df = load_ehr_data(args.data)
        wd = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
        X, y, _, groups = build_xy_longitudinal(df, windows_days=wd)
    else:
        df = load_data(args.data)
        X, y, _, groups = build_xy_tabular(df)

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise SystemExit("Need at least 2 patients for GroupKFold.")
    n_splits = max(2, min(args.splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    aucs, briers = [], []
    for tr, va in gkf.split(X, y, groups):
        m = fit_model(args.model, X.iloc[tr], y.iloc[tr])
        pr = m.predict_proba(X.iloc[va])[:, 1]
        yt = y.iloc[va].to_numpy()
        if len(np.unique(yt)) >= 2:
            aucs.append(float(roc_auc_score(yt, pr)))
        briers.append(float(brier_score_loss(yt, pr)))

    out = {
        "data_path": str(args.data.resolve()),
        "format": args.format,
        "model_kind": args.model,
        "n_splits_effective": n_splits,
        "roc_auc_mean": float(np.nanmean(aucs)) if aucs else None,
        "roc_auc_std": float(np.nanstd(aucs)) if aucs else None,
        "roc_auc_folds": aucs,
        "brier_mean": float(np.mean(briers)) if briers else None,
        "brier_std": float(np.std(briers)) if briers else None,
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
