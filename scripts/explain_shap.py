#!/usr/bin/env python3
"""Generate SHAP summary plot from trained artifact and data (rebuilds same split as training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib

from explainability.shap_explainer import explain_model
from training.reproduce_split import split_train_test_from_artifact
from utils.config import MODEL_PATH, REPORTS_DIR


def main():
    ap = argparse.ArgumentParser(
        description="SHAP summary on the same train/holdout split as training (from artifact meta)."
    )
    ap.add_argument("--artifact", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "shap_summary.png")
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    model = art["model"]
    fe = art.get("feature_engineering", {})
    if not fe.get("data_path"):
        raise SystemExit("Artifact missing feature_engineering.data_path; re-train with current training.train.")

    X_train, X_test, _, _, _, _ = split_train_test_from_artifact(art)
    explain_model(model, X_train, X_test, plot_path=args.out, random_state=int(fe.get("random_state", 42)))
    print(f"Wrote {args.out} (split_method={fe.get('split_method', 'random_row')})")


if __name__ == "__main__":
    main()
