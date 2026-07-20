#!/usr/bin/env python3
"""
Paper Experiments E and F — synthetic integrity contrasts (not clinical).

E: Controlled post-index glucose injection into an otherwise integrity-aware matrix.
F: Naive notebook path (no truncate) vs integrity truncate — audit + multi-seed AUC.

  PYTHONPATH=. python scripts/run_paper_integrity_experiments.py \\
    --data data/raw/paper_synthetic_cohort.csv \\
    --out-dir reports/paper
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from feature_engineering.cohort_integrity import (
    audit_temporal_integrity,
    horizon_labels,
    resolve_index_times,
    truncate_events_to_index,
)
from feature_engineering.multi_window import merge_multi_window_features
from preprocessing.cleaning import clean_longitudinal_ehr
from preprocessing.ehr_loader import load_ehr_data
from training.train import fit_model
from utils.json_safe import json_safe


WINDOWS = (7, 30, 180)


def _patient_label_frame(
    df: pd.DataFrame,
    index_times: pd.Series,
    *,
    horizon_days: int,
    label_col: str,
) -> pd.DataFrame:
    y = horizon_labels(
        df,
        index_times,
        horizon_days=horizon_days,
        patient_col="patient_id",
        time_col="timestamp",
        label_col=label_col,
    )
    out = y.rename(label_col).reset_index()
    out.columns = ["patient_id", label_col]
    return out


def _features_from_events(
    feat_df: pd.DataFrame,
    index_times: pd.Series,
    *,
    inclusive: bool,
) -> pd.DataFrame:
    return merge_multi_window_features(
        feat_df,
        windows_days=WINDOWS,
        patient_col="patient_id",
        time_col="timestamp",
        index_times=index_times,
        inclusive=inclusive,
    )


def _holdout_logreg_auc(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    *,
    seed: int,
    test_size: float = 0.2,
) -> tuple[float, int]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    model = fit_model("logreg", X.iloc[train_idx], y.iloc[train_idx])
    prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    yt = y.iloc[test_idx].to_numpy().astype(int)
    if len(np.unique(yt)) < 2:
        return float("nan"), int(len(yt))
    return float(roc_auc_score(yt, prob)), int(len(yt))


def run_experiment_e(
    df: pd.DataFrame,
    index_times: pd.Series,
    y_df: pd.DataFrame,
    *,
    seed: int,
    label_col: str,
    inclusive: bool,
) -> dict:
    """Integrity-aware matrix vs same matrix + mean post-index glucose."""
    trunc = truncate_events_to_index(
        df,
        index_times,
        patient_col="patient_id",
        time_col="timestamp",
        inclusive=inclusive,
    )
    feat_clean = _features_from_events(trunc, index_times, inclusive=inclusive)
    merged = feat_clean.merge(y_df, on="patient_id", how="inner")
    groups = merged["patient_id"].to_numpy()
    y = merged[label_col].astype(int)
    X_clean = merged.drop(columns=["patient_id", label_col]).fillna(0.0)

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    post_rows = []
    for pid, t_index in index_times.items():
        g = work.loc[work["patient_id"] == pid]
        post = g[g["timestamp"] > t_index]
        mean_g = float(post["glucose"].mean()) if len(post) and "glucose" in post.columns else 0.0
        post_rows.append({"patient_id": pid, "leak_post_index_mean_glucose": mean_g})
    leak_map = {r["patient_id"]: r["leak_post_index_mean_glucose"] for r in post_rows}
    X_leaky = X_clean.copy()
    X_leaky["leak_post_index_mean_glucose"] = (
        merged["patient_id"].map(leak_map).astype(float).fillna(0.0).to_numpy()
    )

    auc_clean, n_test = _holdout_logreg_auc(X_clean, y, groups, seed=seed)
    auc_leak, _ = _holdout_logreg_auc(X_leaky, y, groups, seed=seed)

    audit_trunc = audit_temporal_integrity(
        df,
        index_times,
        horizon_days=365,
        feature_inclusive=inclusive,
        feature_df=trunc,
        label_col=label_col,
    )
    audit_raw = audit_temporal_integrity(
        df,
        index_times,
        horizon_days=365,
        feature_inclusive=inclusive,
        feature_df=df,
        label_col=label_col,
    )

    return {
        "integrity_aware_logreg_roc_auc": round(auc_clean, 4),
        "with_injected_post_index_glucose_feature_roc_auc": round(auc_leak, 4),
        "delta_auc": round(auc_leak - auc_clean, 4),
        "n_test": n_test,
        "n_patients": int(len(index_times)),
        "post_index_events_available_in_raw": int(audit_raw["feature_events_after_index"]),
        "post_index_events_in_truncated_feature_table": int(
            audit_trunc["feature_events_after_index"]
        ),
        "audit_truncated_passed": bool(audit_trunc["passed"]),
        "audit_truncated_feature_events_after_index": int(
            audit_trunc["feature_events_after_index"]
        ),
        "audit_raw_as_features_passed": bool(audit_raw["passed"]),
        "audit_raw_as_features_events_after_index": int(
            audit_raw["feature_events_after_index"]
        ),
        "seed": seed,
        "note": (
            "Controlled synthetic injection: append mean post-index glucose to an "
            "otherwise integrity-aware matrix. Audit fails when feature_df retains "
            "post-index rows. Discrimination change is illustrative only."
        ),
    }


def run_experiment_f(
    df: pd.DataFrame,
    index_times: pd.Series,
    y_df: pd.DataFrame,
    *,
    seeds: list[int],
    label_col: str,
    inclusive: bool,
) -> dict:
    """Naive (no truncate) vs integrity truncate — multi-seed AUC + audit."""
    trunc = truncate_events_to_index(
        df,
        index_times,
        patient_col="patient_id",
        time_col="timestamp",
        inclusive=inclusive,
    )
    feat_integrity = _features_from_events(trunc, index_times, inclusive=inclusive)
    # Naive notebook path: aggregate on the full event table (includes post-index).
    feat_naive = _features_from_events(df, index_times, inclusive=inclusive)

    def _xy(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        merged = feat.merge(y_df, on="patient_id", how="inner")
        groups = merged["patient_id"].to_numpy()
        y = merged[label_col].astype(int)
        X = merged.drop(columns=["patient_id", label_col]).fillna(0.0)
        return X, y, groups

    X_i, y_i, g_i = _xy(feat_integrity)
    X_n, y_n, g_n = _xy(feat_naive)

    audit_integrity = audit_temporal_integrity(
        df,
        index_times,
        horizon_days=365,
        feature_inclusive=inclusive,
        feature_df=trunc,
        label_col=label_col,
    )
    audit_naive = audit_temporal_integrity(
        df,
        index_times,
        horizon_days=365,
        feature_inclusive=inclusive,
        feature_df=df,
        label_col=label_col,
    )

    per_seed = []
    for seed in seeds:
        auc_i, n_test = _holdout_logreg_auc(X_i, y_i, g_i, seed=seed)
        auc_n, _ = _holdout_logreg_auc(X_n, y_n, g_n, seed=seed)
        per_seed.append(
            {
                "seed": seed,
                "n_test": n_test,
                "integrity_roc_auc": round(auc_i, 4),
                "naive_no_truncate_roc_auc": round(auc_n, 4),
                "delta_auc_naive_minus_integrity": round(auc_n - auc_i, 4),
            }
        )

    deltas = [r["delta_auc_naive_minus_integrity"] for r in per_seed]
    auc_i_list = [r["integrity_roc_auc"] for r in per_seed]
    auc_n_list = [r["naive_no_truncate_roc_auc"] for r in per_seed]

    return {
        "n_patients": int(len(index_times)),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "windows": list(WINDOWS),
        "audit_integrity_passed": bool(audit_integrity["passed"]),
        "audit_integrity_feature_events_after_index": int(
            audit_integrity["feature_events_after_index"]
        ),
        "audit_naive_passed": bool(audit_naive["passed"]),
        "audit_naive_feature_events_after_index": int(
            audit_naive["feature_events_after_index"]
        ),
        "integrity_roc_auc_mean": round(float(np.mean(auc_i_list)), 4),
        "integrity_roc_auc_std": round(float(np.std(auc_i_list, ddof=1)), 4)
        if len(auc_i_list) > 1
        else 0.0,
        "naive_roc_auc_mean": round(float(np.mean(auc_n_list)), 4),
        "naive_roc_auc_std": round(float(np.std(auc_n_list, ddof=1)), 4)
        if len(auc_n_list) > 1
        else 0.0,
        "delta_auc_mean": round(float(np.mean(deltas)), 4),
        "delta_auc_std": round(float(np.std(deltas, ddof=1)), 4) if len(deltas) > 1 else 0.0,
        "per_seed": per_seed,
        "note": (
            "Experiment F: naive multi-window aggregation without truncating to "
            "index_time versus the integrity path. Same horizon labels. Multi-seed "
            "patient-disjoint hold-outs. Illustrative software verification only."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper Experiments E and F")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/paper"))
    ap.add_argument("--paper-tables", type=Path, default=Path("research-paper/tables"))
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--index-time-col", default="index_time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--f-seeds",
        default="42,43,44,45,46",
        help="Comma-separated seeds for Experiment F multi-seed contrast",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = args.paper_tables
    tables.mkdir(parents=True, exist_ok=True)

    df = clean_longitudinal_ehr(load_ehr_data(args.data))
    label_col = "label" if "label" in df.columns else "chronic_disease"
    index_times = resolve_index_times(
        df,
        patient_col="patient_id",
        time_col="timestamp",
        index_time_col=args.index_time_col,
        index_strategy="column",
    )
    y_df = _patient_label_frame(
        df, index_times, horizon_days=args.horizon_days, label_col=label_col
    )
    prevalence = float(y_df[label_col].mean())

    e = run_experiment_e(
        df,
        index_times,
        y_df,
        seed=args.seed,
        label_col=label_col,
        inclusive=True,
    )
    e["label_prevalence"] = round(prevalence, 4)
    e_path = out_dir / "experiment_e_leakage_contrast.json"
    e_path.write_text(json.dumps(json_safe(e), indent=2) + "\n")
    (tables / "experiment_e_leakage_contrast.json").write_text(
        json.dumps(json_safe(e), indent=2) + "\n"
    )

    f_seeds = [int(x.strip()) for x in args.f_seeds.split(",") if x.strip()]
    f = run_experiment_f(
        df,
        index_times,
        y_df,
        seeds=f_seeds,
        label_col=label_col,
        inclusive=True,
    )
    f["label_prevalence"] = round(prevalence, 4)
    f_path = out_dir / "experiment_f_naive_vs_integrity.json"
    f_path.write_text(json.dumps(json_safe(f), indent=2) + "\n")
    (tables / "experiment_f_naive_vs_integrity.json").write_text(
        json.dumps(json_safe(f), indent=2) + "\n"
    )

    print(f"Wrote {e_path}")
    print(
        f"  E: integrity AUC={e['integrity_aware_logreg_roc_auc']} "
        f"leaky={e['with_injected_post_index_glucose_feature_roc_auc']} "
        f"audit_raw_fail events={e['audit_raw_as_features_events_after_index']}"
    )
    print(f"Wrote {f_path}")
    print(
        f"  F: integrity mean AUC={f['integrity_roc_auc_mean']}±{f['integrity_roc_auc_std']} "
        f"naive={f['naive_roc_auc_mean']}±{f['naive_roc_auc_std']} "
        f"delta={f['delta_auc_mean']}±{f['delta_auc_std']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
