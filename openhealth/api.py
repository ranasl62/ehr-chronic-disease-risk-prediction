"""Thin facade over existing train / evaluate / explain / artifact paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.config import MODEL_PATH, PROJECT_ROOT, REPORTS_DIR


def train(
    *,
    data_path: str | Path,
    model_kind: str = "logreg",
    data_format: str = "longitudinal",
    out: str | Path | None = None,
    task: str | Path | None = None,
    **kwargs: Any,
):
    """Train a model; optional ``task`` YAML path or id fills defaults."""
    from openhealth.task_spec import load_task
    from training.train import run_training

    params: dict[str, Any] = {
        "data_path": data_path,
        "model_kind": model_kind,
        "data_format": data_format,
        "model_path": out or MODEL_PATH,
    }
    if task is not None:
        spec = load_task(task)
        filled = spec.to_train_params(str(data_path) if data_path else None)
        for k, v in filled.items():
            if k == "task_id":
                continue
            if v is not None:
                params[k] = v
        params["data_path"] = data_path or filled["data_path"]
        params["model_path"] = out or MODEL_PATH
    params.update(kwargs)
    params.pop("task_id", None)
    windows = params.get("windows_days")
    if windows is not None and not isinstance(windows, tuple):
        params["windows_days"] = tuple(windows)
    return run_training(**params)


def evaluate(artifact_path: str | Path | None = None) -> dict[str, Any]:
    """Return metrics from evaluation_report.json or rebuild from artifact."""
    from utils.eval_report import load_evaluation_report_safe

    ev = load_evaluation_report_safe()
    if ev:
        return ev
    path = Path(artifact_path or MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError("No evaluation report and no model artifact")
    return {"artifact": str(path), "metrics": None}


def predict(features: dict[str, float], artifact_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(artifact_path or MODEL_PATH)
    art = joblib.load(path)
    cols = art["feature_columns"]
    missing = [c for c in cols if c not in features]
    if missing:
        raise ValueError(
            "missing required features: "
            + ", ".join(missing[:12])
            + ("…" if len(missing) > 12 else "")
        )
    bad = [c for c in cols if not isinstance(features[c], (int, float)) or not np.isfinite(float(features[c]))]
    if bad:
        raise ValueError("non-finite feature values: " + ", ".join(bad[:8]))
    row = pd.DataFrame([{c: float(features[c]) for c in cols}])[cols]
    model = art["model"]
    proba = float(model.predict_proba(row)[0, 1])
    return {"risk_probability": proba, "feature_columns": cols}


def explain(artifact_path: str | Path | None = None, out: str | Path | None = None) -> Path:
    from explainability.shap_explainer import explain_model
    from training.reproduce_split import split_train_test_from_artifact
    from utils.report_images import require_valid_report_png

    path = Path(artifact_path or MODEL_PATH)
    art = joblib.load(path)
    plot = Path(out or (REPORTS_DIR / "shap_summary.png"))
    if not plot.is_absolute():
        plot = PROJECT_ROOT / plot
    X_train, X_test, _, _, _, _ = split_train_test_from_artifact(art)
    fe = art.get("feature_engineering") or {}
    explain_model(
        art["model"],
        X_train,
        X_test,
        plot_path=plot,
        random_state=int(fe.get("random_state", 42)),
    )
    return require_valid_report_png(plot, label="SHAP summary PNG")


def save_model(artifact: dict[str, Any], path: str | Path | None = None) -> Path:
    out = Path(path or MODEL_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    return out
