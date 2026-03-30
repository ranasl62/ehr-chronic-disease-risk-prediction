---
name: ehr-chronic-risk
description: EHR chronic disease risk — train, API, Streamlit, SHAP, schema. Use for longitudinal features, train CLI, FastAPI, Docker.
---

# EHR chronic risk — compact skill

## Read order
1. `README.md` 2) This file 3) `ehr-ml-pipeline.mdc` *or* `ehr-api-dashboard.mdc` only if editing that area.

## Commands
| Goal | Command |
|------|---------|
| Train longitudinal | `python -m training.train --format longitudinal --data data/raw/ehr_data.csv --split-by-patient` |
| Calibrate | add `--calibrate` |
| Normalize external CSV | `python scripts/normalize_longitudinal_csv.py IN.csv -o data/processed/out.csv` |
| Leakage / split QA | `python scripts/leakage_audit.py --artifact model.pkl` or `--data ... --split-by-patient` |
| SHAP (holdout = train split) | `python scripts/explain_shap.py --artifact model.pkl` |
| Fairness table (holdout) | `python scripts/fairness_report.py --artifact model.pkl --subgroups groups.csv --group-column ...` |
| Docker smoke | `bash scripts/docker_smoke.sh` |
| API | `uvicorn api.main:app --reload` — `GET /v1/meta`, `GET /v1/model/schema` |
| Test | `PYTHONPATH=. pytest tests/ -q` |

## Files
`training/train.py`, `feature_engineering/multi_window.py`, `preprocessing/canonical_schema.py`, `api/main.py`, `dashboard/app.py`, `explainability/shap_explainer.py`, `docs/data_sources_and_schema.md`, `docs/external_validation.md`.

## Non-negotiables
- Temporal features ≤ index time; patient splits for eval when `--split-by-patient`.
- `feature_columns` in artifact must match API/dashboard; list-of-str in `model.pkl`.
- Human author: `AUTHORS.md`.
