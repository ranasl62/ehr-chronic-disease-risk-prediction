"""Task YAML loader — define prediction targets without editing core code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT

TASKS_DIR = PROJECT_ROOT / "tasks"


@dataclass
class TaskSpec:
    id: str
    name: str
    description: str = ""
    target_column: str | None = None
    horizon_days: int | None = None
    index_strategy: str = "last_event"
    index_time_col: str | None = None
    feature_inclusive: bool = True
    windows_days: list[int] = field(default_factory=lambda: [7, 30, 180])
    window_days: int = 180
    data_format: str = "longitudinal"
    suggested_path: str | None = None
    model_kind: str = "logreg"
    calibrate: bool = False
    split_by_patient: bool = True
    temporal_split: bool = False
    metrics: list[str] = field(default_factory=lambda: ["roc_auc", "pr_auc", "brier", "ece"])
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_train_params(self, data_path: str | None = None) -> dict[str, Any]:
        path = data_path or self.suggested_path
        if not path:
            raise ValueError(f"Task {self.id}: no data_path or suggested_path")
        return {
            "data_path": path,
            "data_format": self.data_format,
            "model_kind": self.model_kind,
            "calibrate": self.calibrate,
            "split_by_patient": self.split_by_patient and not self.temporal_split,
            "temporal_split": self.temporal_split,
            "windows_days": list(self.windows_days) if self.windows_days else None,
            "window_days": self.window_days,
            "horizon_days": self.horizon_days,
            "index_strategy": self.index_strategy,
            "index_time_col": self.index_time_col,
            "feature_inclusive": self.feature_inclusive,
            "label_col": self.target_column,
            "task_id": self.id,
        }

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "target_column": self.target_column,
            "horizon_days": self.horizon_days,
            "index_strategy": self.index_strategy,
            "index_time_col": self.index_time_col,
            "feature_inclusive": self.feature_inclusive,
            "windows_days": list(self.windows_days),
            "window_days": self.window_days,
            "data_format": self.data_format,
            "suggested_path": self.suggested_path,
            "model_kind": self.model_kind,
            "calibrate": self.calibrate,
            "split_by_patient": self.split_by_patient,
            "temporal_split": self.temporal_split,
            "metrics": list(self.metrics),
            "source_path": self.source_path,
        }


def _require_yaml():
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Task YAML requires PyYAML (pip install pyyaml)") from e
    return yaml


def load_task(path: str | Path) -> TaskSpec:
    yaml = _require_yaml()
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        # allow bare id: diabetes → tasks/diabetes.yaml
        cand = TASKS_DIR / f"{path}.yaml" if not str(path).endswith(".yaml") else TASKS_DIR / Path(path).name
        if cand.is_file():
            p = cand
        else:
            raise FileNotFoundError(f"Task not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid task YAML (expected mapping): {p}")

    task = raw.get("task") or {}
    target = raw.get("target") or {}
    prediction = raw.get("prediction") or raw.get("prediction_window") or {}
    features = raw.get("features") or {}
    data = raw.get("data") or {}
    training = raw.get("training") or {}
    evaluation = raw.get("evaluation") or {}

    tid = str(task.get("id") or p.stem)
    name = str(task.get("name") or tid)

    # Flexible target shapes: string column or {column: ...} or {diabetes_event: true}
    target_column = None
    if isinstance(target, str):
        target_column = target
    elif isinstance(target, dict):
        target_column = target.get("column") or target.get("name")
        if target_column is None and len(target) == 1:
            target_column = next(iter(target.keys()))

    horizon = prediction.get("horizon_days")
    if horizon is None and "365_days" in str(prediction.get("window", "")):
        horizon = 365
    if isinstance(horizon, str) and horizon.endswith("_days"):
        horizon = int(horizon.replace("_days", ""))

    windows = features.get("windows_days") or features.get("windows")
    if windows is None:
        windows = [7, 30, 180]
    parsed_windows: list[int] = []
    for w in windows:
        if isinstance(w, int):
            parsed_windows.append(w)
        elif isinstance(w, str):
            parsed_windows.append(int(w.replace("_days", "").replace("d", "")))

    return TaskSpec(
        id=tid,
        name=name,
        description=str(task.get("description") or ""),
        target_column=str(target_column) if target_column else None,
        horizon_days=int(horizon) if horizon is not None else None,
        index_strategy=str(prediction.get("index_strategy") or "last_event"),
        index_time_col=prediction.get("index_time_col"),
        feature_inclusive=bool(prediction.get("feature_inclusive", True)),
        windows_days=parsed_windows,
        window_days=int(features.get("window_days") or 180),
        data_format=str(data.get("format") or "longitudinal"),
        suggested_path=data.get("suggested_path") or data.get("path"),
        model_kind=str(training.get("model_kind") or training.get("model") or "logreg"),
        calibrate=bool(training.get("calibrate", False)),
        split_by_patient=bool(training.get("split_by_patient", True)),
        temporal_split=bool(training.get("temporal_split", False)),
        metrics=list(evaluation.get("metrics") or ["roc_auc", "pr_auc", "brier", "ece"]),
        source_path=str(p.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in p.resolve().parents else str(p),
        raw=raw,
    )


def list_tasks(directory: Path | None = None) -> list[TaskSpec]:
    d = directory or TASKS_DIR
    if not d.is_dir():
        return []
    out: list[TaskSpec] = []
    for p in sorted(d.glob("*.yaml")):
        try:
            out.append(load_task(p))
        except Exception:
            continue
    return out
