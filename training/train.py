import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
):
    df = clean_longitudinal_ehr(df)
    lc = label_col or ("label" if "label" in df.columns else "chronic_disease")
    if lc not in df.columns:
        raise ValueError(f"Label column not found: expected 'label' or 'chronic_disease'.")

    y_df = df.groupby("patient_id", as_index=False)[lc].max()
    if windows_days:
        feat = merge_multi_window_features(
            df,
            windows_days=windows_days,
            patient_col="patient_id",
            time_col="timestamp",
        )
    else:
        feat = create_time_window_features(df, window_days=window_days)
    merged = feat.merge(y_df, on="patient_id")
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
):
    data_path = Path(
        data_path
        or (DEFAULT_EHR_LONGITUDINAL_CSV if data_format == "longitudinal" else DEFAULT_RAW_CSV)
    )
    model_path = Path(model_path or MODEL_PATH)

    if data_format == "longitudinal":
        df = load_ehr_data(data_path)
        X, y, feature_columns, groups = build_xy_longitudinal(
            df,
            window_days=window_days,
            windows_days=windows_days,
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
    args = p.parse_args()
    windows_days: tuple[int, ...] | None = None
    if args.format == "longitudinal":
        if args.windows is not None and args.windows.strip() == "":
            windows_days = None
        else:
            wspec = (args.windows or "7,30,180").strip()
            windows_days = tuple(int(x.strip()) for x in wspec.split(",") if x.strip())

    run_training(
        data_path=args.data,
        model_path=args.out,
        model_kind=args.model,
        data_format=args.format,
        window_days=args.window_days,
        windows_days=windows_days,
        calibrate=args.calibrate,
        calibration_plot_path=args.calibration_plot,
        skip_calibration_plot=args.no_calibration_plot,
        split_by_patient=args.split_by_patient,
        temporal_split=args.temporal_split,
        bootstrap_samples=args.bootstrap_samples if args.bootstrap_samples > 0 else None,
        ece_bins=args.ece_bins,
    )


if __name__ == "__main__":
    main()
