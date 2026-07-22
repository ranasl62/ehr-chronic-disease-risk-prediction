"""Multi-format dataset ingest and downloadable research packs."""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.data_quality import assert_no_blocking_errors, summarize_csv
from utils.config import PROJECT_ROOT, REPORTS_DIR

UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
_SAFE = re.compile(r"^[a-zA-Z0-9._\-]+$")

_DELETABLE_ROOTS = (UPLOADS_DIR, DEMO_DIR, RAW_DIR)

_ZIP_ALLOW = [
    "evaluation_report.json",
    "feature_importance.json",
    "training_manifest.json",
    "leakage_audit.json",
    "calibration_holdout.png",
    "shap_summary.png",
    "cv_group_metrics.json",
    "model_comparison.json",
    "fairness_report.json",
    "hpo_report.json",
    "threshold_operating_points.json",
]


def ensure_uploads() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def _is_under_deletable_root(path: Path) -> bool:
    return any(
        path.is_relative_to(root.resolve())
        for root in _DELETABLE_ROOTS
    )


def delete_dataset_file(path_str: str) -> dict[str, Any]:
    """Idempotently delete a dataset file under an approved data root."""
    project_root = PROJECT_ROOT.resolve()
    supplied = Path(path_str)
    candidate = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
    # Normalize ``..`` without resolving symlinks; a symlink's destination is
    # validated separately before it can be removed.
    candidate = Path(os.path.abspath(candidate))

    try:
        candidate.relative_to(project_root)
    except ValueError as e:
        raise ValueError("path must be under project root") from e

    if not _is_under_deletable_root(candidate):
        raise ValueError("Only files under data/uploads, data/demo, or data/raw can be deleted")

    if candidate.name in {".gitkeep", "README.md"} or candidate.name.startswith("."):
        raise ValueError("Refusing to delete protected file")
    if candidate.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("Only .csv or companion .json dataset files can be deleted")

    relative_path = str(candidate.relative_to(project_root))
    if not candidate.exists() and not candidate.is_symlink():
        return {
            "deleted": True,
            "already_absent": True,
            "path": relative_path,
            "requested": path_str,
        }

    target = candidate.resolve(strict=False)
    if candidate.is_symlink():
        if target.is_file() and _is_under_deletable_root(target):
            if target.name in {".gitkeep", "README.md"} or target.name.startswith("."):
                raise ValueError("Refusing to delete protected file")
            if target.suffix.lower() not in {".csv", ".json"}:
                raise ValueError("Only .csv or companion .json dataset files can be deleted")
        candidate.unlink()
        # A broken or external target is never followed. Removing the approved
        # symlink itself is still safe and satisfies the delete request.
        if not target.is_file() or not _is_under_deletable_root(target):
            return {
                "deleted": True,
                "already_absent": False,
                "path": relative_path,
                "requested": path_str,
            }
    else:
        if not candidate.is_file():
            raise ValueError("Only files can be deleted")
        if not _is_under_deletable_root(target):
            raise ValueError("Only files under data/uploads, data/demo, or data/raw can be deleted")

    if not _is_under_deletable_root(target):
        # This guard protects against a race that changes a symlink target.
        raise ValueError("Only files under data/uploads, data/demo, or data/raw can be deleted")
    if target.name in {".gitkeep", "README.md"} or target.name.startswith("."):
        raise ValueError("Refusing to delete protected file")
    if target.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("Only .csv or companion .json dataset files can be deleted")

    target.unlink()
    if target.suffix.lower() == ".csv":
        companion = target.parent / f"{target.stem}.meta.json"
        if companion.is_file() or companion.is_symlink():
            companion.unlink()

    return {
        "deleted": True,
        "already_absent": False,
        "path": relative_path,
        "requested": path_str,
    }


def _save_dataframe(df: pd.DataFrame, name: str) -> dict[str, Any]:
    ensure_uploads()
    if not name.lower().endswith(".csv"):
        name = f"{Path(name).stem}.csv"
    if not _SAFE.match(name):
        raise ValueError("Unsafe filename")
    dest = UPLOADS_DIR / name
    # Prefer longitudinal if timestamp present
    fmt = "longitudinal" if "timestamp" in df.columns else "tabular"
    if "patient_id" not in df.columns:
        raise ValueError("CSV must include patient_id")
    df.to_csv(dest, index=False)
    issues = summarize_csv(dest, data_format=fmt)
    try:
        assert_no_blocking_errors(issues)
    except ValueError:
        # try other format
        other = "tabular" if fmt == "longitudinal" else "longitudinal"
        issues = summarize_csv(dest, data_format=other)
        assert_no_blocking_errors(issues)
        fmt = other
    return {
        "id": f"upload:{name}",
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "format": fmt,
        "bytes": dest.stat().st_size,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "issues": issues,
    }


def dataframe_from_upload_bytes(filename: str, data: bytes) -> pd.DataFrame:
    lower = filename.lower()
    bio = io.BytesIO(data)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    if lower.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict) and "records" in obj:
            return pd.DataFrame(obj["records"])
        if isinstance(obj, dict) and "data" in obj:
            return pd.DataFrame(obj["data"])
        raise ValueError("JSON must be a list of records or {records|data: [...]}")
    if lower.endswith(".tsv"):
        return pd.read_csv(bio, sep="\t")
    if lower.endswith(".csv"):
        return pd.read_csv(bio)
    raise ValueError("Supported: .csv, .tsv, .json, .xlsx, .xls")


def import_file_bytes(filename: str, data: bytes) -> dict[str, Any]:
    from openhealth.schema_map import enrich_upload_with_aliases

    df = dataframe_from_upload_bytes(filename, data)
    df = enrich_upload_with_aliases(df)
    stem = Path(filename).stem
    meta = _save_dataframe(df, f"{stem}.csv")
    meta["source_type"] = "byo"
    return meta


def import_form_rows(rows: list[dict[str, Any]], name: str = "form_import.csv") -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be non-empty")
    df = pd.DataFrame(rows)
    return _save_dataframe(df, name)


def import_sql(
    sql: str,
    *,
    connection_url: str | None = None,
    name: str = "sql_import.csv",
) -> dict[str, Any]:
    """
    Run a read-only SELECT against DATABASE_URL (env) or explicit connection_url.
    Blocks obvious mutating statements.
    """
    import os

    text = sql.strip().rstrip(";")
    low = text.lower()
    if not low.startswith("select") and not low.startswith("with"):
        raise ValueError("Only SELECT / WITH … SELECT queries are allowed")
    forbidden = (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " truncate ", " grant ")
    padded = f" {low} "
    if any(tok in padded for tok in forbidden):
        raise ValueError("Mutating SQL is not allowed")
    url = (connection_url or os.environ.get("DATABASE_URL", "")).strip()
    if not url:
        raise ValueError(
            "Set DATABASE_URL (e.g. postgresql://… or sqlite:///…) or pass connection_url"
        )
    try:
        from sqlalchemy import create_engine, text as sql_text
    except ImportError as e:
        raise ValueError("SQL import requires sqlalchemy (pip install sqlalchemy)") from e
    engine = create_engine(url)
    with engine.connect() as conn:
        df = pd.read_sql(sql_text(text), conn)
    return _save_dataframe(df, name)


def profile_dataset(
    path: Path,
    *,
    age_band: str | None = None,
    label: str | None = None,
    patient_id: str | None = None,
    max_cohort: int = 400,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else (
        "chronic_disease" if "chronic_disease" in df.columns else None
    )
    work = df.copy()
    if "age" in work.columns:
        ages = pd.to_numeric(work["age"], errors="coerce")
        bins = [0, 40, 50, 60, 70, 120]
        band_labels = ["lt40", "40_49", "50_59", "60_69", "ge70"]
        work["_age_band"] = pd.cut(ages, bins=bins, labels=band_labels, right=False).astype(str)
    else:
        work["_age_band"] = None
    if age_band and "_age_band" in work.columns:
        work = work[work["_age_band"] == age_band]
    if label is not None and label_col and label_col in work.columns:
        work = work[work[label_col].astype(str) == str(label)]
    if patient_id is not None and "patient_id" in work.columns:
        work = work[work["patient_id"].astype(str) == str(patient_id)]

    out: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.resolve().parents or path.resolve().parent == PROJECT_ROOT else str(path),
        "n_rows": int(len(work)),
        "n_columns": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "n_patients": int(work["patient_id"].nunique()) if "patient_id" in work.columns and len(work) else (int(df["patient_id"].nunique()) if "patient_id" in df.columns else None),
        "filters": {"age_band": age_band, "label": label, "patient_id": patient_id},
    }
    if label_col and len(work):
        vc = work.groupby("patient_id")[label_col].max() if "patient_id" in work.columns else work[label_col]
        counts = vc.value_counts(dropna=False).to_dict()
        out["label_column"] = label_col
        out["label_counts"] = {str(k): int(v) for k, v in counts.items()}
    elif label_col:
        out["label_column"] = label_col
        out["label_counts"] = {}
    if "age" in work.columns and len(work):
        band_vc = work["_age_band"].value_counts(dropna=False).to_dict()
        out["age_band_counts"] = {str(k): int(v) for k, v in band_vc.items() if k not in ("nan", "None", None)}
    miss_src = work if len(work) else df
    miss = (miss_src.isna().mean() * 100).sort_values(ascending=False).head(12)
    out["missing_pct"] = {str(k): round(float(v), 2) for k, v in miss.items() if not str(k).startswith("_")}
    numeric_preview = {}
    for c in ("glucose", "blood_pressure", "cholesterol", "lab_value", "vital_signs"):
        if c in miss_src.columns:
            s = pd.to_numeric(miss_src[c], errors="coerce")
            numeric_preview[c] = {
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() else None,
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
            }
    out["numeric_preview"] = numeric_preview
    if "timestamp" in miss_src.columns:
        ts = pd.to_datetime(miss_src["timestamp"], errors="coerce")
        out["time_span"] = {
            "min": str(ts.min()) if ts.notna().any() else None,
            "max": str(ts.max()) if ts.notna().any() else None,
        }
    # Patient-level cohort sample for UI filters / tables
    cohort: list[dict[str, Any]] = []
    if "patient_id" in work.columns and len(work):
        gcols: dict[str, Any] = {}
        if label_col:
            gcols[label_col] = "max"
        if "age" in work.columns:
            gcols["age"] = "last"
        if "sex" in work.columns:
            gcols["sex"] = "last"
        if "glucose" in work.columns:
            gcols["glucose"] = "mean"
        if gcols:
            agg = work.groupby("patient_id", as_index=False).agg(gcols)
        else:
            agg = work[["patient_id"]].drop_duplicates()
        if "_age_band" in work.columns:
            bands = work.groupby("patient_id")["_age_band"].last().reset_index()
            agg = agg.merge(bands, on="patient_id", how="left")
        for _, row in agg.head(max_cohort).iterrows():
            cohort.append(
                {
                    "patient_id": str(row.get("patient_id")),
                    "age": float(row["age"]) if "age" in row and pd.notna(row.get("age")) else None,
                    "age_band": str(row["_age_band"]) if "_age_band" in row and pd.notna(row.get("_age_band")) else None,
                    "sex": str(row["sex"]) if "sex" in row and pd.notna(row.get("sex")) else None,
                    "label": str(row[label_col]) if label_col and label_col in row and pd.notna(row.get(label_col)) else None,
                    "glucose_mean": float(row["glucose"]) if "glucose" in row and pd.notna(row.get("glucose")) else None,
                }
            )
    out["cohort_rows"] = cohort
    out["filter_options"] = {
        "age_bands": sorted({c["age_band"] for c in cohort if c.get("age_band")}),
        "labels": sorted({c["label"] for c in cohort if c.get("label") is not None}),
        "sexes": sorted({c["sex"] for c in cohort if c.get("sex")}),
    }
    return out


def _audit_pass_fail_counts(audit: dict[str, Any] | None) -> tuple[int, int]:
    """Count boolean integrity flags in a leakage audit tree (passed / failed)."""
    if not isinstance(audit, dict):
        return 0, 0
    passed = 0
    failed = 0

    def walk(obj: Any) -> None:
        nonlocal passed, failed
        if isinstance(obj, dict):
            if "passed" in obj and isinstance(obj["passed"], bool):
                if obj["passed"]:
                    passed += 1
                else:
                    failed += 1
            if "patient_disjoint_train_test" in obj and isinstance(
                obj["patient_disjoint_train_test"], bool
            ):
                if obj["patient_disjoint_train_test"]:
                    passed += 1
                else:
                    failed += 1
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(audit)
    return passed, failed


def build_methods_markdown() -> str:
    """Short research-only methods note from active report artifacts (no invented claims)."""
    manifest: dict[str, Any] = {}
    mp = REPORTS_DIR / "training_manifest.json"
    if mp.is_file():
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = raw
        except Exception:
            manifest = {}

    evaluation: dict[str, Any] = {}
    ep = REPORTS_DIR / "evaluation_report.json"
    if ep.is_file():
        try:
            raw = json.loads(ep.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                evaluation = raw
        except Exception:
            evaluation = {}

    leakage: dict[str, Any] | None = None
    lp = REPORTS_DIR / "leakage_audit.json"
    if lp.is_file():
        try:
            raw = json.loads(lp.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                leakage = raw
        except Exception:
            leakage = None

    meta = evaluation.get("meta") if isinstance(evaluation.get("meta"), dict) else {}
    fe = meta.get("feature_engineering") if isinstance(meta.get("feature_engineering"), dict) else {}
    tm = meta.get("training_manifest") if isinstance(meta.get("training_manifest"), dict) else {}

    model_kind = (
        manifest.get("model_kind")
        or meta.get("model_kind")
        or tm.get("model_kind")
        or "unknown"
    )
    calibrated = manifest.get("calibrated")
    if calibrated is None:
        calibrated = meta.get("calibrated")
    if calibrated is None:
        calibrated = tm.get("calibrated")

    split_method = (
        manifest.get("split_method")
        or fe.get("split_method")
        or tm.get("split_method")
        or ("temporal" if manifest.get("temporal_split") else "unknown")
    )
    horizon = fe.get("horizon_days")
    if horizon is None and leakage:
        horizon = leakage.get("horizon_days")
        ti = leakage.get("temporal_integrity") if isinstance(leakage.get("temporal_integrity"), dict) else {}
        if horizon is None:
            horizon = ti.get("horizon_days")
    index_strategy = fe.get("index_strategy")
    if index_strategy is None and leakage:
        index_strategy = leakage.get("index_strategy")
        ti = leakage.get("temporal_integrity") if isinstance(leakage.get("temporal_integrity"), dict) else {}
        if index_strategy is None:
            index_strategy = ti.get("index_strategy")

    data_hash = manifest.get("data_sha256") or tm.get("data_sha256")
    data_path = manifest.get("data_path") or fe.get("data_path")
    task_id = manifest.get("task_id") or meta.get("task_id") or "not recorded"
    windows = fe.get("windows_days") or fe.get("window_days")

    pass_n, fail_n = _audit_pass_fail_counts(leakage)

    lines = [
        "# Methods note (auto-generated)",
        "",
        "> Research and education artifact only — not a clinical protocol, not FDA-cleared,",
        "> and not intended for patient care. Values below reflect the active workspace reports,",
        "> not a claim of external validity.",
        "",
        "## Run configuration",
        "",
        f"- **Task id:** `{task_id}`",
        f"- **Model:** `{model_kind}`",
        f"- **Isotonic calibration:** `{calibrated}`",
        f"- **Split:** `{split_method}`"
        + (f" (temporal_split=`{manifest.get('temporal_split')}`)" if "temporal_split" in manifest else ""),
        f"- **Index strategy:** `{index_strategy if index_strategy is not None else 'not recorded'}`",
        f"- **Horizon days:** `{horizon if horizon is not None else 'not recorded'}`",
        f"- **Feature windows:** `{windows if windows is not None else 'not recorded'}`",
        f"- **Data path:** `{data_path if data_path is not None else 'not recorded'}`",
        f"- **Data SHA-256:** `{data_hash if data_hash is not None else 'not recorded'}`",
        "",
        "## Leakage audit summary",
        "",
    ]
    if leakage is None:
        lines.append("- No `leakage_audit.json` present in reports.")
    else:
        lines.append(f"- Boolean integrity flags counted: **pass={pass_n}**, **fail={fail_n}**.")
        ti = leakage.get("temporal_integrity") if isinstance(leakage.get("temporal_integrity"), dict) else None
        if ti is not None:
            lines.append(
                f"- Temporal integrity `passed`: `{ti.get('passed')}` "
                f"(post-index feature events: `{ti.get('feature_events_after_index')}`)."
            )
        if "patient_disjoint_train_test" in leakage:
            lines.append(
                f"- Patient-disjoint train/test: `{leakage.get('patient_disjoint_train_test')}`."
            )
    lines.extend(
        [
            "",
            "## Honesty limits",
            "",
            "- Metrics and figures in the results pack are hold-out / workspace reports only.",
            "- Do not treat this note as a complete CONSORT/TRIPOD methods section.",
            "- Re-run training and leakage audit after any data or pipeline change.",
            "",
        ]
    )
    return "\n".join(lines)


def build_results_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in _ZIP_ALLOW:
            p = REPORTS_DIR / name
            if p.is_file():
                zf.write(p, arcname=f"reports/{name}")
        methods_md = build_methods_markdown()
        zf.writestr("reports/methods.md", methods_md)
        # Include model card + quickstart if present
        for rel in (
            "docs/model_card.md",
            "docs/researcher_quickstart.md",
            "CITATION.cff",
            "AUTHORS.md",
            "LIMITATIONS.md",
        ):
            p = PROJECT_ROOT / rel
            if p.is_file():
                zf.write(p, arcname=rel)
        # Always include a short limitations excerpt
        lim = PROJECT_ROOT / "LIMITATIONS.md"
        if lim.is_file():
            zf.writestr(
                "LIMITATIONS_EXCERPT.md",
                "\n".join(lim.read_text(encoding="utf-8").splitlines()[:40]) + "\n…\n",
            )
        manifest = {
            "pack": "ehr-risk-research-results",
            "disclaimer": "Research and education artifacts only — not intended for patient care.",
            "files": [n for n in _ZIP_ALLOW if (REPORTS_DIR / n).is_file()] + ["methods.md"],
        }
        zf.writestr("README_PACK.json", json.dumps(manifest, indent=2))
    return buf.getvalue()
