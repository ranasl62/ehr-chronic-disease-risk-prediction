"""Workspace config store — YAML round-trip for Config Center."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "workspace.yaml"

_DEFAULT: dict[str, Any] = {
    "persona": "researcher",  # researcher | clinical_research
    "active_task_id": None,
    "windows_days": [7, 30, 180],
    "horizon_days": None,
    "index_strategy": "last_event",
    "index_time_col": None,
    "model_kind": "logreg",
    "compare_models": ["logreg", "random_forest", "xgboost"],
    "calibrate": False,
    "split_by_patient": True,
    "temporal_split": False,
    "feature_inclusive": True,
    "require_api_key": False,
    "disclaimer_ack": False,
    "active_run_id": None,
    "data_mode": "synthetic",  # synthetic | real
    # Researcher workbench UI preferences (also mirrored in browser localStorage)
    "ui": {
        "density": "comfortable",  # comfortable | compact
        "theme": "forest",  # forest | slate | sand
        "analytics_view": "split",  # charts | tables | split
        "chart_animation": True,
        "table_page_size": 10,
        "top_n_features": 15,
        "show_label_chart": True,
        "show_age_chart": True,
        "show_missing_chart": True,
        "show_numeric_chart": True,
        "show_metric_chart": True,
        "show_importance_chart": True,
        "show_compare_chart": True,
    },
}

_VALID_PERSONAS = {"researcher", "clinical_research"}
_VALID_MODELS = {"logreg", "random_forest", "xgboost", "lightgbm"}


def default_config() -> dict[str, Any]:
    return deepcopy(_DEFAULT)


def load_config(path: Path | None = None) -> dict[str, Any]:
    p = path or CONFIG_PATH
    cfg = default_config()
    if not p.is_file():
        return cfg
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for workspace config") from e
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("workspace.yaml must be a mapping")
    cfg.update({k: v for k, v in raw.items() if k in _DEFAULT or k in raw})
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    persona = cfg.get("persona", "researcher")
    if persona not in _VALID_PERSONAS:
        raise ValueError(f"Invalid persona: {persona}")
    mk = cfg.get("model_kind")
    if mk and mk not in _VALID_MODELS:
        raise ValueError(f"Invalid model_kind: {mk}")
    for m in cfg.get("compare_models") or []:
        if m not in _VALID_MODELS:
            raise ValueError(f"Invalid compare model: {m}")


def save_config(cfg: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    import yaml

    validate_config(cfg)
    p = path or CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = default_config()
    merged.update(cfg)
    validate_config(merged)
    p.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return merged


def effective_train_params(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    c = cfg or load_config()
    return {
        "windows_days": c.get("windows_days"),
        "horizon_days": c.get("horizon_days"),
        "index_strategy": c.get("index_strategy", "last_event"),
        "index_time_col": c.get("index_time_col"),
        "model_kind": c.get("model_kind", "logreg"),
        "calibrate": bool(c.get("calibrate")),
        "split_by_patient": bool(c.get("split_by_patient", True)),
        "temporal_split": bool(c.get("temporal_split")),
        "feature_inclusive": bool(c.get("feature_inclusive", True)),
        "task_id": c.get("active_task_id"),
        "persona": c.get("persona", "researcher"),
        "disclaimer_ack": bool(c.get("disclaimer_ack")),
    }
