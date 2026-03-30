import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
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
DEFAULT_RAW_CSV = DATA_RAW / "sample_ehr.csv"
DEFAULT_EHR_LONGITUDINAL_CSV = DATA_RAW / "ehr_data.csv"
