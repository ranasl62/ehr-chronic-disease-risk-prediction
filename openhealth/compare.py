"""Multi-model comparison (classical models) — not full AutoML."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from training.reporting import build_evaluation_report, save_json
from training.train import run_training
from utils.config import MODEL_PATH, PROJECT_ROOT, REPORTS_DIR
from utils.json_safe import json_safe


def available_models() -> list[str]:
    models = ["logreg", "random_forest", "xgboost"]
    try:
        import lightgbm  # noqa: F401

        models.append("lightgbm")
    except Exception:
        pass
    return models


def _score(metrics: dict[str, Any]) -> float:
    auc = metrics.get("roc_auc")
    if auc is None or (isinstance(auc, float) and auc != auc):
        return -1.0
    return float(auc)


def compare_models(
    *,
    data_path: str | Path,
    data_format: str = "longitudinal",
    models: list[str] | None = None,
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
    out_dir: Path | None = None,
    promote_best: bool = True,
) -> dict[str, Any]:
    """
    Train each model, write per-model artifacts under reports/compare/,
    rank by ROC-AUC, optionally promote best → model.pkl.
    """
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    out_dir = out_dir or (REPORTS_DIR / "compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = models or available_models()
    windows = tuple(windows_days) if windows_days else None
    rows: list[dict[str, Any]] = []

    for kind in kinds:
        tag = kind
        model_path = out_dir / f"model_{tag}.pkl"
        cal_plot = out_dir / f"calibration_{tag}.png"
        _, X_test, y_test, artifact = run_training(
            data_path=data_path,
            model_path=model_path,
            model_kind=kind,
            data_format=data_format,
            window_days=window_days,
            windows_days=windows,
            calibrate=calibrate,
            calibration_plot_path=cal_plot,
            skip_calibration_plot=False,
            split_by_patient=split_by_patient,
            temporal_split=temporal_split,
            horizon_days=horizon_days,
            index_strategy=index_strategy,
            index_time_col=index_time_col,
            feature_inclusive=feature_inclusive,
            label_col=label_col,
        )
        eval_report = build_evaluation_report(
            artifact["model"],
            X_test,
            y_test,
        )
        metrics = eval_report.get("metrics") or {}
        save_json(out_dir / f"eval_{tag}.json", eval_report)
        try:
            rel_model = str(model_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_model = str(model_path)
        rows.append(
            {
                "model": kind,
                "metrics": metrics,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "brier": metrics.get("brier"),
                "ece": metrics.get("ece"),
                "calibrated": calibrate,
                "model_path": rel_model,
                "selected": False,
            }
        )

    rows.sort(key=lambda r: _score(r.get("metrics") or {}), reverse=True)
    if rows:
        rows[0]["selected"] = True
        best = rows[0]
        if promote_best:
            src = Path(best["model_path"])
            if not src.is_absolute():
                src = PROJECT_ROOT / src
            shutil.copy2(src, MODEL_PATH)
            best_eval = out_dir / f"eval_{best['model']}.json"
            if best_eval.is_file():
                shutil.copy2(best_eval, REPORTS_DIR / "evaluation_report.json")

    try:
        data_rel = str(data_path.relative_to(PROJECT_ROOT))
    except ValueError:
        data_rel = str(data_path)
    summary = {
        "data_path": data_rel,
        "models_compared": kinds,
        "calibrate": calibrate,
        "ranking_metric": "roc_auc",
        "comparison": rows,
        "selected_model": rows[0]["model"] if rows else None,
        "disclaimer": "For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future.",
    }
    out_json = REPORTS_DIR / "model_comparison.json"
    out_json.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    save_json(out_dir / "summary.json", summary)
    return summary
