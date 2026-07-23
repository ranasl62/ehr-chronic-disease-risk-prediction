"""Score a frozen model artifact on a second dataset (research external validation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from training.evaluate import evaluate_binary
from training.train import build_xy_longitudinal, build_xy_tabular
from utils.config import PROJECT_ROOT, REPORTS_DIR
from utils.json_safe import json_safe


def _align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(
            "External dataset missing required feature columns from the trained contract: "
            + ", ".join(missing[:20])
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )
    return X.reindex(columns=feature_columns)


def external_validate(
    *,
    artifact_path: Path,
    data_path: Path,
    data_format: str = "longitudinal",
    label_col: str | None = None,
) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"model artifact missing: {artifact_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"external data not found: {data_path}")

    art = joblib.load(artifact_path)
    model = art["model"]
    feature_columns = list(art.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("artifact missing feature_columns")

    fe = art.get("feature_engineering") or {}
    fmt = data_format or fe.get("format") or "longitudinal"
    lc = label_col or fe.get("label_col")

    if fmt == "longitudinal":
        from preprocessing.ehr_loader import load_ehr_data

        df = load_ehr_data(data_path)
        if lc is None:
            lc = "label" if "label" in df.columns else "chronic_disease"
        windows = fe.get("windows_days")
        kwargs = {
            "horizon_days": fe.get("horizon_days"),
            "index_strategy": fe.get("index_strategy") or "last_event",
            "index_time_col": fe.get("index_time_col"),
            "feature_inclusive": True
            if fe.get("feature_inclusive") is None
            else bool(fe.get("feature_inclusive")),
            "label_col": lc,
        }
        if windows:
            X, y, _, _ = build_xy_longitudinal(df, windows_days=tuple(windows), **kwargs)
        else:
            X, y, _, _ = build_xy_longitudinal(
                df, window_days=int(fe.get("window_days") or 180), **kwargs
            )
    else:
        from preprocessing.ehr_loader import load_data

        df = load_data(data_path)
        X, y, _, _ = build_xy_tabular(df)

    X_aligned = _align_features(X, feature_columns)
    metrics = evaluate_binary(model, X_aligned, y)
    # Drop bulky sklearn text report for JSON pack
    metrics_clean = {k: v for k, v in metrics.items() if k != "report"}

    return {
        "kind": "external_validation",
        "artifact": str(artifact_path.resolve()),
        "data_path": str(data_path.resolve()),
        "data_format": fmt,
        "n_samples": int(len(X_aligned)),
        "n_positive": int(np.asarray(y).sum()) if len(y) else 0,
        "label_prevalence": float(np.asarray(y).mean()) if len(y) else None,
        "metrics": metrics_clean,
        "disclaimer": "Research hold-out on a second CSV — not multi-site clinical validation.",
    }


def write_external_validation_report(
    report: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shared = REPORTS_DIR / "external_validation_report.json"
    payload = json.dumps(json_safe(report), indent=2)
    shared.write_text(payload, encoding="utf-8")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "external_validation_report.json").write_text(payload, encoding="utf-8")
    return shared
