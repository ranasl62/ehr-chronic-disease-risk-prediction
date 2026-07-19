#!/usr/bin/env python3
"""
Run paper experiment matrix → reports/paper/.

Baselines × calibration ablation × window ablation, with bootstrap ROC-AUC CIs
when the hold-out is large enough.

  PYTHONPATH=. python scripts/run_paper_experiments.py \\
    --data data/raw/paper_synthetic_cohort.csv \\
    --horizon-days 365 --index-strategy column --index-time-col index_time \\
    --split-by-patient --out-dir reports/paper
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

from training.bootstrap_metrics import bootstrap_roc_auc_ci
from training.reporting import build_evaluation_report, save_json
from training.train import run_training
from utils.json_safe import json_safe


def _metrics_from_eval(eval_report: dict) -> dict:
    m = eval_report.get("metrics") or eval_report
    return {
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "brier": m.get("brier"),
        "ece": m.get("ece"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
    }


def _try_models() -> list[str]:
    models = ["logreg", "random_forest", "xgboost"]
    try:
        import lightgbm  # noqa: F401

        models.append("lightgbm")
    except Exception:
        pass
    return models


def run_one(
    *,
    data: Path,
    out_dir: Path,
    model: str,
    calibrate: bool,
    windows: tuple[int, ...] | None,
    window_days: int,
    horizon_days: int,
    index_strategy: str,
    index_time_col: str | None,
    feature_inclusive: bool,
    split_by_patient: bool,
    temporal_split: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    wlabel = "-".join(map(str, windows)) if windows else str(window_days)
    tag = f"{model}_{'cal' if calibrate else 'raw'}_w{wlabel}"
    model_path = out_dir / f"model_{tag}.pkl"
    cal_plot = out_dir / f"calibration_{tag}.png"
    _, X_test, y_test, artifact = run_training(
        data_path=data,
        model_path=model_path,
        model_kind=model,
        data_format="longitudinal",
        window_days=window_days,
        windows_days=windows,
        calibrate=calibrate,
        calibration_plot_path=cal_plot,
        skip_calibration_plot=False,
        split_by_patient=split_by_patient,
        temporal_split=temporal_split,
        bootstrap_samples=bootstrap_samples if bootstrap_samples > 0 else None,
        horizon_days=horizon_days,
        index_strategy=index_strategy,
        index_time_col=index_time_col,
        feature_inclusive=feature_inclusive,
        random_state=seed,
    )
    model_obj = artifact["model"]
    eval_report = build_evaluation_report(
        model_obj,
        X_test,
        y_test,
        meta={
            "tag": tag,
            "model_kind": model,
            "calibrated": calibrate,
            "windows": list(windows) if windows else [window_days],
        },
    )
    row: dict = {
        "tag": tag,
        "model": model,
        "calibrated": calibrate,
        "windows": list(windows) if windows else [window_days],
        "n_test": int(len(y_test)),
        "metrics": _metrics_from_eval(eval_report),
        "model_path": str(model_path),
    }
    if bootstrap_samples > 0 and len(y_test) >= 5 and y_test.nunique() > 1:
        prob = model_obj.predict_proba(X_test)[:, 1]
        row["bootstrap_roc_auc"] = bootstrap_roc_auc_ci(
            np.asarray(y_test),
            prob,
            n_bootstrap=bootstrap_samples,
            random_state=seed,
        )
    if len(y_test) >= 5 and y_test.nunique() > 1:
        prob = model_obj.predict_proba(X_test)[:, 1]
        y = np.asarray(y_test).astype(int)
        prevalence = float(y.mean())
        nbs = []
        for thr in (0.1, 0.2, 0.3, 0.5):
            pred = (prob >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            n = len(y)
            nb = (tp / n) - (fp / n) * (thr / max(1e-9, (1.0 - thr)))
            nbs.append({"threshold": thr, "net_benefit": float(nb)})
        row["decision_curve"] = {"prevalence": prevalence, "points": nbs}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper experiment matrix")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/paper"))
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--index-strategy", default="column")
    ap.add_argument("--index-time-col", default="index_time")
    ap.add_argument("--feature-exclusive", action="store_true")
    ap.add_argument("--split-by-patient", action="store_true", default=True)
    ap.add_argument("--temporal-split", action="store_true")
    ap.add_argument("--bootstrap-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only logreg + xgboost, multi-window, cal/raw (faster CI).",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["logreg", "xgboost"] if args.quick else _try_models()
    # (windows_days tuple or None, single window_days)
    specs: list[tuple[tuple[int, ...] | None, int]] = [
        ((7, 30, 180), 180),
        (None, 180),
    ]
    if args.quick:
        specs = [((7, 30, 180), 180)]

    rows: list[dict] = []
    for model in models:
        for calibrate in (False, True):
            for windows, window_days in specs:
                try:
                    row = run_one(
                        data=args.data,
                        out_dir=out_dir,
                        model=model,
                        calibrate=calibrate,
                        windows=windows,
                        window_days=window_days,
                        horizon_days=args.horizon_days,
                        index_strategy=args.index_strategy,
                        index_time_col=args.index_time_col,
                        feature_inclusive=not args.feature_exclusive,
                        split_by_patient=args.split_by_patient and not args.temporal_split,
                        temporal_split=args.temporal_split,
                        bootstrap_samples=args.bootstrap_samples,
                        seed=args.seed,
                    )
                    rows.append(row)
                    print(f"OK {row['tag']} roc_auc={row['metrics'].get('roc_auc')}")
                except Exception as exc:
                    rows.append({"tag": f"{model}_cal{calibrate}_{windows or window_days}", "error": str(exc)})
                    print(f"FAIL {model} cal={calibrate} windows={windows}: {exc}")

    summary = {
        "data": str(args.data.resolve()),
        "horizon_days": args.horizon_days,
        "index_strategy": args.index_strategy,
        "index_time_col": args.index_time_col,
        "n_runs": len(rows),
        "runs": rows,
        "note": "Synthetic or local cohort metrics — software verification only, not clinical performance.",
    }
    save_json(out_dir / "experiment_summary.json", json_safe(summary))

    flat = []
    for r in rows:
        if "error" in r:
            flat.append({"tag": r.get("tag"), "error": r["error"]})
            continue
        m = r.get("metrics") or {}
        flat.append(
            {
                "tag": r["tag"],
                "model": r["model"],
                "calibrated": r["calibrated"],
                "windows": ",".join(map(str, r["windows"])),
                "n_test": r["n_test"],
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "brier": m.get("brier"),
                "ece": m.get("ece"),
            }
        )
    pd.DataFrame(flat).to_csv(out_dir / "results_table.csv", index=False)
    print(f"Wrote {out_dir / 'experiment_summary.json'} and results_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
