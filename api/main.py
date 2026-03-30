import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.cors import CORSMiddleware

from explainability.explanation import build_patient_explanation
from inference.predict import load_artifact, predict_row
from inference.validation import validate_feature_dict
from utils.config import MODEL_PATH
from utils.eval_report import evaluation_aligned_with_manifest, load_evaluation_report_safe
from utils.json_safe import json_safe

from api.middleware import RequestContextMiddleware, configure_api_logging
from api.production_middleware import BodySizeLimitMiddleware, RateLimitMiddleware
from api.security import require_api_key_if_configured

log = logging.getLogger("ehr_api")

AuthDep = Depends(require_api_key_if_configured)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_api_logging()
    if os.environ.get("API_KEY", "").strip():
        log.warning("API_KEY is set; prediction routes require X-API-Key header.")
    else:
        log.info("API_KEY unset; prediction routes are open (development mode).")
    yield


app = FastAPI(
    title="EHR Chronic Disease Risk API",
    version="0.6.1",
    description="Production-style inference with optional API key, request IDs, and SHAP-ready artifacts.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestContextMiddleware)

_max_body = int(os.environ.get("MAX_BODY_BYTES", "262144"))
_rpm = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "0"))
if _rpm > 0:
    app.add_middleware(RateLimitMiddleware, per_minute=_rpm)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=_max_body)


class PatientFeaturesV1(BaseModel):
    """Fixed schema for the tabular demo (sample_ehr.csv) pipeline."""

    age: float = Field(..., ge=0, le=120)
    glucose: float = Field(..., ge=0)
    blood_pressure: float = Field(..., ge=0)
    cholesterol: float = Field(..., ge=0)


class FeaturesDict(BaseModel):
    """Feature vector aligned to `GET /v1/model/schema`."""

    features: dict[str, float]

    @field_validator("features")
    @classmethod
    def non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("features must be non-empty")
        return v


class PredictFeaturesBody(FeaturesDict):
    """Prediction request with optional patient-level explanation (default on)."""

    include_explanation: bool = True


@lru_cache(maxsize=1)
def get_artifact():
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail=f"No model at {MODEL_PATH}. Run: python -m training.train",
        )
    return load_artifact(MODEL_PATH)


def artifact_dep() -> dict[str, Any]:
    """Injectable model artifact (override in tests via `app.dependency_overrides`)."""
    return get_artifact()


ArtifactDep = Depends(artifact_dep)


def _predict_row_http(features: dict[str, float], artifact: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_row(features, artifact=artifact)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/")
def home():
    return {
        "message": "EHR chronic disease risk prediction API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/v1/ready",
        "meta": "/v1/meta",
        "schema": "/v1/model/schema",
        "metrics": "/v1/model/metrics",
        "predict": "/v1/predict",
        "explain": "/explain",
        "security": "Set API_KEY env + X-API-Key header for protected routes when deployed.",
    }


@app.get("/health")
def health():
    """Liveness: process up and model file on disk (fast — use for k8s liveness)."""
    present = Path(MODEL_PATH).exists()
    return {
        "status": "ok" if present else "degraded",
        "model_file_present": present,
        "model_path": str(MODEL_PATH),
    }


@app.get("/v1/ready")
def readiness():
    """Readiness: artifact loads successfully (use for load balancers; heavier than /health)."""
    present = Path(MODEL_PATH).exists()
    if not present:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "model_file_missing",
                "model_path": str(MODEL_PATH),
            },
        )
    try:
        load_artifact(MODEL_PATH)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "model_load_failed",
                "model_path": str(MODEL_PATH),
                "error": str(e),
            },
        )
    return {"ready": True, "model_path": str(MODEL_PATH)}


@app.get("/v1/meta")
def api_meta():
    """Deployment metadata and governance (no PHI). Safe for public load balancers."""
    return {
        "api_version": app.version,
        "model_path_configured": str(MODEL_PATH),
        "model_file_present": Path(MODEL_PATH).exists(),
        "clinical_use": "prohibited_without_validation",
        "description": (
            "Research and population-health prototyping only. Not FDA-cleared; "
            "not for diagnosis or treatment. Institutional policy and IRB govern access to real PHI."
        ),
        "documentation": {
            "data_schema": "docs/data_sources_and_schema.md",
            "study_protocol": "docs/study_protocol.md",
            "external_validation": "docs/external_validation.md",
        },
        "health": {"liveness": "/health", "readiness": "/v1/ready"},
    }


@app.get("/v1/model/schema")
def model_schema(_: bool = AuthDep, art: dict[str, Any] = ArtifactDep):
    return {
        "feature_columns": art["feature_columns"],
        "model_kind": art.get("model_kind", "unknown"),
        "calibrated": art.get("calibrated", False),
        "feature_engineering": art.get("feature_engineering", {}),
        "input_stats": art.get("input_stats"),
    }


@app.get("/v1/model/metrics")
def model_metrics(_: bool = AuthDep, art: dict[str, Any] = ArtifactDep):
    """Hold-out metrics from `reports/evaluation_report.json` if present (no PHI)."""
    ev = load_evaluation_report_safe()
    if not ev:
        raise HTTPException(
            status_code=503,
            detail="evaluation_report.json missing. Run: python -m training.train",
        )
    meta = ev.get("meta") or {}
    aligned = evaluation_aligned_with_manifest(art.get("training_manifest"), meta)
    payload = {
        "evaluation_aligned_with_loaded_artifact": aligned,
        "generated_at_utc": ev.get("generated_at_utc"),
        "threshold": ev.get("threshold"),
        "metrics": ev.get("metrics"),
        "meta_summary": {
            "split_method": (meta.get("feature_engineering") or {}).get("split_method"),
            "psi_train_vs_test_predicted_prob": meta.get("psi_train_vs_test_predicted_prob"),
            "bootstrap_roc_auc": meta.get("bootstrap_roc_auc"),
        },
    }
    return json_safe(payload)


@app.post("/predict")
def predict_legacy(
    features: PatientFeaturesV1,
    _: bool = AuthDep,
    artifact: dict[str, Any] = ArtifactDep,
):
    """Backward-compatible body: flat age, glucose, blood_pressure, cholesterol."""
    data = features.model_dump()
    out = _predict_row_http(data, artifact)
    return {
        "risk_score": out["risk_probability"],
        "risk_probability": out["risk_probability"],
        "risk_class": out["risk_class"],
        "risk_level": out["risk_level"],
    }


@app.post("/v1/predict")
def predict_v1(
    body: PredictFeaturesBody,
    _: bool = AuthDep,
    artifact: dict[str, Any] = ArtifactDep,
):
    """Production path: feature dict must match artifact `feature_columns` exactly."""
    out = _predict_row_http(body.features, artifact)
    resp: dict[str, Any] = {
        "risk_score": out["risk_probability"],
        "risk_probability": out["risk_probability"],
        "risk_class": out["risk_class"],
        "risk_level": out["risk_level"],
    }
    if body.include_explanation:
        resp["explanation"] = build_patient_explanation(artifact, body.features)
    return resp


@app.post("/explain")
def explain_endpoint(
    body: FeaturesDict,
    _: bool = AuthDep,
    artifact: dict[str, Any] = ArtifactDep,
):
    """Patient-level interpretability JSON (SHAP or structured fallback)."""
    try:
        validate_feature_dict(
            body.features,
            required_columns=artifact["feature_columns"],
            allow_extra=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return build_patient_explanation(artifact, body.features)


@app.post("/predict/raw")
def predict_raw(
    patient: dict[str, Any],
    include_explanation: bool = Query(default=True),
    _: bool = AuthDep,
    artifact: dict[str, Any] = ArtifactDep,
):
    """
    Flat JSON object feature_name -> number. Optional query `include_explanation=false`.
    """
    try:
        flat = {k: float(v) for k, v in patient.items()}
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="All feature values must be numeric") from e
    feat = {k: flat[k] for k in artifact["feature_columns"]}
    out = _predict_row_http(feat, artifact)
    resp: dict[str, Any] = {
        "risk_score": out["risk_probability"],
        "risk_probability": out["risk_probability"],
        "risk_level": out["risk_level"],
    }
    if include_explanation:
        resp["explanation"] = build_patient_explanation(artifact, feat)
    return resp
