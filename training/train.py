import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering.cohort_integrity import (
    horizon_labels,
    resolve_index_times,
    truncate_events_to_index,
)
from feature_engineering.multi_window import merge_multi_window_features
from feature_engineering.patient_features import create_features
from feature_engineering.time_window_features import create_time_window_features
from models.baseline_logreg import train_model as train_logreg
from models.calibration import calibrate_model
from models.random_forest_model import train_rf
from preprocessing.cleaning import basic_clean, clean_longitudinal_ehr
from preprocessing.ehr_loader import load_data, load_ehr_data
from training.bootstrap_metrics import bootstrap_roc_auc_ci
from training.eval_plots import save_calibration_curve_plot
from training.evaluate import print_lead_time_summary, print_metrics
from training.manifest import build_training_manifest
from training.splits import (
    group_train_test_split,
    temporal_patient_train_test_split,
    train_test_split_safe,
)
from inference.validation import build_input_stats_frame
from training.reporting import (
    build_evaluation_report,
    feature_importance_from_model,
    save_json,
)
from utils.config import (
    DEFAULT_EHR_LONGITUDINAL_CSV,
    DEFAULT_RAW_CSV,
    EVALUATION_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TRAINING_MANIFEST_PATH,
    resolve_training_data_path,
)


def build_xy_tabular(df: pd.DataFrame):
    df = basic_clean(df)
    features = create_features(df)
    y_df = df.groupby("patient_id", as_index=False)["chronic_disease"].max()
    merged = features.merge(y_df, on="patient_id")
    groups = merged["patient_id"].to_numpy()
    y = merged["chronic_disease"].astype(int)
    X = merged.drop(columns=["patient_id", "chronic_disease"])
    return X, y, list(X.columns), groups


def build_xy_longitudinal(
    df: pd.DataFrame,
    window_days: int = 180,
    label_col: str | None = None,
    *,
    windows_days: tuple[int, ...] | None = None,
    horizon_days: int | None = None,
    index_strategy: str = "last_event",
    index_time_col: str | None = None,
    feature_inclusive: bool = True,
):
    """
    Build patient-level feature matrix and labels.

    When ``horizon_days`` is set, labels use only events in (index, index+H]
    and features are built from events truncated to the index time.
    """
    df = clean_longitudinal_ehr(df)
    candidates = [c for c in (label_col, "label", "chronic_disease") if c]
    lc = next((c for c in candidates if c in df.columns), None)
    if lc is None:
        raise ValueError(
            "Label column not found: expected 'label' or 'chronic_disease'"
            + (f" (requested {label_col!r})" if label_col else "")
            + "."
        )

    strategy = index_strategy
    if horizon_days is not None and strategy == "last_event" and index_time_col is None:
        # Without an explicit index column, before_last avoids using the outcome
        # row both as feature anchor and as the sole label signal.
        strategy = "before_last"

    index_times = resolve_index_times(
        df,
        patient_col="patient_id",
        time_col="timestamp",
        index_time_col=index_time_col,
        index_strategy=strategy if index_time_col is None else "column",
    )

    if horizon_days is not None:
        y_series = horizon_labels(
            df,
            index_times,
            horizon_days=horizon_days,
            patient_col="patient_id",
            time_col="timestamp",
            label_col=lc,
        )
        y_df = y_series.rename(lc).reset_index()
        y_df.columns = ["patient_id", lc]
        feat_df = truncate_events_to_index(
            df,
            index_times,
            patient_col="patient_id",
            time_col="timestamp",
            inclusive=feature_inclusive,
        )
    else:
        y_df = df.groupby("patient_id", as_index=False)[lc].max()
        feat_df = df

    if windows_days:
        feat = merge_multi_window_features(
            feat_df,
            windows_days=windows_days,
            patient_col="patient_id",
            time_col="timestamp",
            index_times=index_times,
            inclusive=feature_inclusive,
        )
    else:
        feat = create_time_window_features(
            feat_df,
            window_days=window_days,
            index_times=index_times,
            inclusive=feature_inclusive,
        )
    merged = feat.merge(y_df, on="patient_id", how="inner")
    groups = merged["patient_id"].to_numpy()
    y = merged[lc].astype(int)
    X = merged.drop(columns=["patient_id", lc])
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    return X, y, list(X.columns), groups


def make_estimator(kind: str, *, y_for_imbalance=None):
    if kind == "xgboost":
        from models.xgboost_model import make_xgb_classifier, xgb_scale_pos_weight

        spw = (
            xgb_scale_pos_weight(y_for_imbalance)
            if y_for_imbalance is not None
            else 1.0
        )
        return make_xgb_classifier(scale_pos_weight=spw)
    if kind == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if kind == "lightgbm":
        from models.lightgbm_model import make_lgbm_estimator

        return make_lgbm_estimator()
    raise ValueError(f"Unknown model kind: {kind}")


def fit_model(kind: str, X_train, y_train):
    """Train an uncalibrated estimator (used by `run_training` and CV scripts)."""
    if kind == "xgboost":
        from models.xgboost_model import train_xgb

        return train_xgb(X_train, y_train)
    if kind == "logreg":
        return train_logreg(X_train, y_train)
    if kind == "random_forest":
        return train_rf(X_train, y_train)
    if kind == "lightgbm":
        from models.lightgbm_model import train_lgb

        return train_lgb(X_train, y_train)
    raise ValueError(f"Unknown model kind: {kind}")


_fit_model = fit_model  # backward compatibility


def run_training(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_kind: str = "xgboost",
    data_format: str = "tabular",
    window_days: int = 180,
    windows_days: tuple[int, ...] | None = None,
    *,
    calibrate: bool = False,
    calibration_plot_path: Path | None = None,
    skip_calibration_plot: bool = False,
    lead_time_days: np.ndarray | pd.Series | None = None,
    split_by_patient: bool = False,
    temporal_split: bool = False,
    bootstrap_samples: int | None = None,
    ece_bins: int = 10,
    horizon_days: int | None = None,
    index_strategy: str = "last_event",
    index_time_col: str | None = None,
    feature_inclusive: bool = True,
    label_col: str | None = None,
):
    data_path = resolve_training_data_path(
        data_path
        or (DEFAULT_EHR_LONGITUDINAL_CSV if data_format == "longitudinal" else DEFAULT_RAW_CSV)
    )
    model_path = Path(model_path or MODEL_PATH)

    if data_format == "longitudinal":
        df = load_ehr_data(data_path)
        X, y, feature_columns, groups = build_xy_longitudinal(
            df,
            window_days=window_days,
            label_col=label_col,
            windows_days=windows_days,
            horizon_days=horizon_days,
            index_strategy=index_strategy,
            index_time_col=index_time_col,
            feature_inclusive=feature_inclusive,
        )
    else:
        df = load_data(data_path)
        X, y, feature_columns, groups = build_xy_tabular(df)

    if temporal_split:
        if data_format != "longitudinal":
            raise ValueError("--temporal-split requires --format longitudinal.")
        df_clean = clean_longitudinal_ehr(df)
        last_e = df_clean.groupby("patient_id")["timestamp"].max()
        X_train, X_test, y_train, y_test = temporal_patient_train_test_split(
            X,
            y,
            groups,
            last_e,
            test_size=test_size,
        )
        split_method = "temporal_patient"
    elif split_by_patient:
        X_train, X_test, y_train, y_test = group_train_test_split(
            X,
            y,
            groups,
            test_size=test_size,
            random_state=random_state,
        )
        split_method = "patient_group"
    else:
        X_train, X_test, y_train, y_test = train_test_split_safe(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        split_method = "random_row"

    shap_bg = X_train.sample(
        n=min(64, len(X_train)),
        random_state=random_state,
    )

    if calibrate:
        est = make_estimator(model_kind, y_for_imbalance=y_train)
        model = calibrate_model(est, X_train, y_train)
    else:
        model = fit_model(model_kind, X_train, y_train)

    if model_kind == "xgboost":
        try:
            from models.xgboost_model import evaluate as xgb_quick_eval

            quick = xgb_quick_eval(model, X_test, y_test)
            auc_s = f"{quick['AUC']:.4f}" if quick["AUC"] == quick["AUC"] else "n/a"
            print(f"XGBoost quick eval: AUC = {auc_s}")
        except ModuleNotFoundError:
            print("XGBoost not installed; skipping quick AUC line.")

    fi = feature_importance_from_model(model, feature_columns)
    input_stats = build_input_stats_frame(X_train)
    fe_meta = {
        "format": data_format,
        "window_days": window_days if data_format == "longitudinal" and not windows_days else None,
        "windows_days": list(windows_days) if windows_days else None,
        "data_path": str(data_path.resolve()),
        "random_state": random_state,
        "test_size": test_size,
        "split_method": split_method,
        "horizon_days": horizon_days,
        "index_strategy": index_strategy if data_format == "longitudinal" else None,
        "index_time_col": index_time_col,
        "feature_inclusive": feature_inclusive if data_format == "longitudinal" else None,
    }
    manifest = build_training_manifest(
        data_path=data_path,
        model_path=model_path,
        model_kind=model_kind,
        calibrated=bool(calibrate),
        split_method=split_method,
        extra={
            "ece_bins": ece_bins,
            "bootstrap_samples": bootstrap_samples,
            "temporal_split": temporal_split,
        },
    )
    save_json(TRAINING_MANIFEST_PATH, manifest)
    manifest_slim = {
        k: manifest[k]
        for k in (
            "generated_at_utc",
            "data_sha256",
            "git_revision",
            "split_method",
            "model_kind",
            "calibrated",
        )
        if k in manifest
    }
    artifact = {
        "model": model,
        "feature_columns": [str(c) for c in list(feature_columns)],
        "model_kind": model_kind,
        "feature_engineering": fe_meta,
        "calibrated": bool(calibrate),
        "feature_importance": fi,
        "shap_background": shap_bg,
        "input_stats": input_stats,
        "training_manifest": manifest_slim,
    }
    joblib.dump(artifact, model_path)

    print_metrics(model, X_test, y_test, ece_bins=ece_bins)

    eval_report = build_evaluation_report(
        model,
        X_test,
        y_test,
        meta={
            "model_path": str(model_path.resolve()),
            "model_kind": model_kind,
            "calibrated": bool(calibrate),
            "feature_engineering": fe_meta,
            "training_manifest_path": str(TRAINING_MANIFEST_PATH.resolve()),
            "training_manifest": manifest_slim,
        },
        ece_bins=ece_bins,
    )
    if bootstrap_samples and len(y_test) >= 5:
        y_prob_h = model.predict_proba(X_test)[:, 1]
        eval_report["meta"]["bootstrap_roc_auc"] = bootstrap_roc_auc_ci(
            np.asarray(y_test),
            y_prob_h,
            n_bootstrap=int(bootstrap_samples),
            random_state=random_state,
        )
    try:
        from monitoring.psi import psi

        ref_p = model.predict_proba(X_train)[:, 1]
        cur_p = model.predict_proba(X_test)[:, 1]
        eval_report["meta"]["psi_train_vs_test_predicted_prob"] = float(psi(ref_p, cur_p))
    except Exception:
        eval_report["meta"]["psi_train_vs_test_predicted_prob"] = None
    save_json(EVALUATION_REPORT_PATH, eval_report)
    save_json(FEATURE_IMPORTANCE_PATH, {"generated_at_utc": eval_report["generated_at_utc"], "importance": fi})
    print(f"Wrote {EVALUATION_REPORT_PATH} and {FEATURE_IMPORTANCE_PATH}")

    if lead_time_days is not None:
        lt = np.asarray(lead_time_days)
        if len(lt) == len(y_test):
            y_prob = model.predict_proba(X_test)[:, 1]
            print_lead_time_summary(lt, y_test, y_prob, threshold=0.5)
        else:
            print(
                "Lead-time array length does not match holdout; "
                "provide one value per test row (same split order)."
            )

    if not skip_calibration_plot:
        out = calibration_plot_path or (REPORTS_DIR / "calibration_holdout.png")
        y_prob = model.predict_proba(X_test)[:, 1]
        save_calibration_curve_plot(y_test, y_prob, out)
        print(f"Calibration curve → {out}")

    print(f"Saved artifact → {model_path}")
    return model, X_test, y_test, artifact


def main():
    p = argparse.ArgumentParser(description="Train chronic disease risk model")
    p.add_argument(
        "--model",
        choices=["xgboost", "logreg", "random_forest", "lightgbm"],
        default="xgboost",
        help="Primary learner (default: xgboost)",
    )
    p.add_argument(
        "--format",
        choices=["tabular", "longitudinal"],
        default="tabular",
        help="tabular: sample_ehr-style; longitudinal: MIMIC-style ehr_data.csv",
    )
    p.add_argument("--data", type=Path, default=None, help="CSV path")
    p.add_argument("--out", type=Path, default=None, help="Output model.pkl path")
    p.add_argument("--window-days", type=int, default=180, help="Single lookback (longitudinal) if --windows unset")
    p.add_argument(
        "--windows",
        type=str,
        default=None,
        help="Longitudinal: comma-separated lookbacks (default 7,30,180). Use empty string for single --window-days only.",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Wrap base estimator in CalibratedClassifierCV (isotonic, internal CV).",
    )
    p.add_argument(
        "--no-calibration-plot",
        action="store_true",
        help="Skip saving hold-out calibration curve PNG.",
    )
    p.add_argument(
        "--calibration-plot",
        type=Path,
        default=None,
        help="Output path for calibration curve (default: reports/calibration_holdout.png).",
    )
    sp = p.add_mutually_exclusive_group()
    sp.add_argument(
        "--split-by-patient",
        action="store_true",
        help="Random group split: no patient_id in both train and test.",
    )
    sp.add_argument(
        "--temporal-split",
        action="store_true",
        help="Longitudinal only: train on patients with earlier last event, test on later (temporal generalization).",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="If >0, percentile bootstrap CI for hold-out ROC-AUC (requires ≥5 test rows).",
    )
    p.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Bins for expected calibration error (ECE) in evaluation_report.json.",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help="Longitudinal: label from events in (index, index+H] only (leakage-safe).",
    )
    p.add_argument(
        "--index-strategy",
        choices=["last_event", "before_last", "column"],
        default="last_event",
        help="How to choose index_time per patient (column requires --index-time-col).",
    )
    p.add_argument(
        "--index-time-col",
        type=str,
        default=None,
        help="Column with per-row index_time when --index-strategy column.",
    )
    p.add_argument(
        "--feature-exclusive",
        action="store_true",
        help="Use half-open feature window [index-W, index) instead of inclusive upper bound.",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task YAML id or path (e.g. diabetes or tasks/diabetes.yaml). Fills train defaults.",
    )
    p.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column override (longitudinal).",
    )
    args = p.parse_args()

    data_path = args.data
    data_format = args.format
    model_kind = args.model
    calibrate = args.calibrate
    split_by_patient = args.split_by_patient
    temporal_split = args.temporal_split
    window_days = args.window_days
    horizon_days = args.horizon_days
    index_strategy = args.index_strategy
    index_time_col = args.index_time_col
    feature_inclusive = not args.feature_exclusive
    label_col = args.label_col
    windows_days: tuple[int, ...] | None = None
    windows_from_task = False

    if args.task:
        from openhealth.task_spec import load_task

        spec = load_task(args.task)
        tp = spec.to_train_params(str(data_path) if data_path else None)
        data_path = Path(tp["data_path"])
        data_format = tp["data_format"]
        model_kind = tp["model_kind"]
        calibrate = bool(tp["calibrate"] or calibrate)
        temporal_split = bool(tp["temporal_split"] or temporal_split)
        split_by_patient = bool(tp["split_by_patient"]) if not temporal_split else False
        if horizon_days is None:
            horizon_days = tp.get("horizon_days")
        if index_time_col is None:
            index_time_col = tp.get("index_time_col")
        if args.index_strategy == "last_event" and tp.get("index_strategy"):
            index_strategy = tp["index_strategy"]
        feature_inclusive = bool(tp.get("feature_inclusive", feature_inclusive))
        label_col = label_col or tp.get("label_col")
        window_days = int(tp.get("window_days") or window_days)
        if tp.get("windows_days"):
            windows_days = tuple(tp["windows_days"])
            windows_from_task = True

    if data_format == "longitudinal" and not windows_from_task:
        if args.windows is not None and args.windows.strip() == "":
            windows_days = None
        else:
            wspec = (args.windows or "7,30,180").strip()
            windows_days = tuple(int(x.strip()) for x in wspec.split(",") if x.strip())

    run_training(
        data_path=data_path,
        model_path=args.out,
        model_kind=model_kind,
        data_format=data_format,
        window_days=window_days,
        windows_days=windows_days,
        calibrate=calibrate,
        calibration_plot_path=args.calibration_plot,
        skip_calibration_plot=args.no_calibration_plot,
        split_by_patient=split_by_patient,
        temporal_split=temporal_split,
        bootstrap_samples=args.bootstrap_samples if args.bootstrap_samples > 0 else None,
        ece_bins=args.ece_bins,
        horizon_days=horizon_days,
        index_strategy=index_strategy,
        index_time_col=index_time_col,
        feature_inclusive=feature_inclusive,
        label_col=label_col,
    )


if __name__ == "__main__":
    main()
