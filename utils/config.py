import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DEMO = PROJECT_ROOT / "data" / "demo"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.json"
TRAINING_MANIFEST_PATH = REPORTS_DIR / "training_manifest.json"


def resolve_model_path() -> Path:
    raw = os.environ.get("MODEL_PATH", "").strip()
    if not raw:
        return PROJECT_ROOT / "model.pkl"
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


MODEL_PATH = resolve_model_path()
DEFAULT_RAW_CSV = DATA_DEMO / "sample_ehr.csv"
DEFAULT_EHR_LONGITUDINAL_CSV = DATA_DEMO / "ehr_data.csv"

_LEGACY_DEMO_PATHS = {
    Path("data/raw/ehr_data.csv"): Path("data/demo/ehr_data.csv"),
    Path("data/raw/sample_ehr.csv"): Path("data/demo/sample_ehr.csv"),
}


def _project_data_relative_path(path: Path) -> Path | None:
    """Return the ``data/...`` portion of a path recorded in another runtime."""
    try:
        data_index = path.parts.index("data")
    except ValueError:
        return None
    relative = Path(*path.parts[data_index:])
    return relative if len(relative.parts) > 1 else None


def resolve_training_data_path(
    path: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    """Resolve current, container-recorded, and legacy demo data paths."""
    root = (project_root or PROJECT_ROOT).resolve()
    requested = Path(path)
    candidate = requested if requested.is_absolute() else root / requested
    if candidate.is_file():
        return candidate.resolve()

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = _project_data_relative_path(requested)
        if relative is None:
            return candidate

    project_candidate = root / relative
    if project_candidate.is_file():
        return project_candidate.resolve()

    fallback = _LEGACY_DEMO_PATHS.get(relative)
    if fallback:
        demo_path = root / fallback
        if demo_path.is_file():
            return demo_path.resolve()
    return candidate
