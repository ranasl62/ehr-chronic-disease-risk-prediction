"""Light hyperparameter grid search (research-scoped; not full AutoML)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.cleaning import clean_longitudinal_ehr
from preprocessing.ehr_loader import load_data, load_ehr_data
from training.evaluate import evaluate_binary
from training.splits import group_train_test_split, temporal_patient_train_test_split, train_test_split_safe
from training.train import build_xy_longitudinal, build_xy_tabular, make_estimator, run_training
from utils.config import PROJECT_ROOT, REPORTS_DIR
from utils.json_safe import json_safe

# Keep grids tiny so jobs finish quickly on demo data.
DEFAULT_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "random_forest": [
        {"max_depth": 4, "n_estimators": 100},
        {"max_depth": 8, "n_estimators": 200},
    ],
    "xgboost": [{"max_depth": 3}, {"max_depth": 5}, {"max_depth": 7}],
    "lightgbm": [{"num_leaves": 15}, {"num_leaves": 31}],
}


def _estimator_with_params(kind: str, params: dict[str, Any], y_train) -> Any:
    if kind == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        C=float(params.get("C", 1.0)),
                    ),
                ),
            ]
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 8)),
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if kind == "xgboost":
        from models.xgboost_model import make_xgb_classifier, xgb_scale_pos_weight

        clf = make_xgb_classifier(scale_pos_weight=xgb_scale_pos_weight(y_train))
        if "max_depth" in params:
            clf.set_params(max_depth=int(params["max_depth"]))
        if "n_estimators" in params:
            clf.set_params(n_estimators=int(params["n_estimators"]))
        return clf
    if kind == "lightgbm":
        from models.lightgbm_model import make_lgbm_estimator

        est = make_lgbm_estimator()
        if "num_leaves" in params:
            est.set_params(num_leaves=int(params["num_leaves"]))
        return est
    return make_estimator(kind, y_for_imbalance=y_train)


def _score(metrics: dict[str, Any]) -> float:
    auc = metrics.get("roc_auc")
    if auc is None or (isinstance(auc, float) and auc != auc):
        return -1.0
    return float(auc)


def run_light_hpo(
    *,
    data_path: str | Path,
    model_kind: str = "logreg",
    data_format: str = "longitudinal",
    calibrate: bool = False,
    split_by_patient: bool = True,
    temporal_split: bool = False,
    windows_days: tuple[int, ...] | list[int] | None = (7, 30, 180),
    window_days: int = 180,
    horizon_days: int | None = None,
    index_strategy: str = "last_event",
    index_time_col: str | None = None,
    feature_inclusive: bool = True,
    label_col: str | None = None,
    grid: list[dict[str, Any]] | None = None,
    promote_best: bool = False,
    max_trials: int = 6,
) -> dict[str, Any]:
    """
    Fit a small hyperparameter grid on the standard train/hold-out split.
    Writes reports/hpo_report.json. Optionally promotes the best config via full retrain.
    """
    from models.calibration import calibrate_model

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"data not found: {data_path}")

    points = list(grid or DEFAULT_GRIDS.get(model_kind) or [{}])
    points = points[: max(1, int(max_trials))]

    if data_format == "longitudinal":
        df = load_ehr_data(data_path)
        X, y, _feature_columns, groups = build_xy_longitudinal(
            df,
            window_days=window_days,
            label_col=label_col,
            windows_days=tuple(windows_days) if windows_days else None,
            horizon_days=horizon_days,
            index_strategy=index_strategy,
            index_time_col=index_time_col,
            feature_inclusive=feature_inclusive,
        )
    else:
        df = load_data(data_path)
        X, y, _feature_columns, groups = build_xy_tabular(df)

    if temporal_split:
        if data_format != "longitudinal":
            raise ValueError("temporal_split requires longitudinal data")
        df_clean = clean_longitudinal_ehr(df)
        last_e = df_clean.groupby("patient_id")["timestamp"].max()
        X_train, X_test, y_train, y_test = temporal_patient_train_test_split(
            X, y, groups, last_e, test_size=0.2
        )
    elif split_by_patient:
        X_train, X_test, y_train, y_test = group_train_test_split(
            X, y, groups, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split_safe(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for i, params in enumerate(points):
        est = _estimator_with_params(model_kind, params, y_train)
        if calibrate:
            model = calibrate_model(est, X_train, y_train)
        else:
            model = est.fit(X_train, y_train)
        metrics = evaluate_binary(model, X_test, y_test)
        row = {
            "trial": i,
            "params": params,
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "brier": metrics.get("brier"),
            "ece": metrics.get("ece"),
            "f1": metrics.get("f1"),
        }
        trials.append(row)
        if best is None or _score(metrics) > _score(best):
            best = {**row, "metrics": {k: v for k, v in metrics.items() if k != "report"}}

    out = {
        "model_kind": model_kind,
        "n_trials": len(trials),
        "trials": trials,
        "best": best,
        "promoted": False,
        "note": "Research-scoped light grid only; not clinical AutoML.",
    }

    if promote_best and best is not None:
        run_training(
            data_path=data_path,
            model_kind=model_kind,
            data_format=data_format,
            window_days=window_days,
            windows_days=tuple(windows_days) if windows_days else None,
            calibrate=calibrate,
            split_by_patient=split_by_patient and not temporal_split,
            temporal_split=temporal_split,
            horizon_days=horizon_days,
            index_strategy=index_strategy,
            index_time_col=index_time_col,
            feature_inclusive=feature_inclusive,
            label_col=label_col,
        )
        out["promoted"] = True
        out["promoted_params"] = best.get("params")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "hpo_report.json"
    path.write_text(json.dumps(json_safe(out), indent=2), encoding="utf-8")
    out["report_path"] = str(path.relative_to(PROJECT_ROOT))
    return out
