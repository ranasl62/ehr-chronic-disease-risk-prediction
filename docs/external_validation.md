# External and temporal validation (closing the generalization gap)

This repo cannot replace **your** second dataset or site, but it supports stronger internal stories than a single random split.

## 1. Temporal patient split (`--temporal-split`)

**Longitudinal only.** Patients are ordered by **last event time** in the cleaned timeline. The earliest \((1 - \text{test\_size})\) fraction → **train**; the latest fraction → **test**.

- **Intent:** mimic “model trained on past cohort, evaluated on patients active later” without row-level leakage.
- **Limit:** still one database; not a true external validator.
- **Commands:**
  ```bash
  python -m training.train --format longitudinal --data your.csv --temporal-split --model xgboost
  PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl
  # If you trained with --temporal-split:
  # PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl --temporal-split
  ```

## 2. Group K-fold (`scripts/group_cv_report.py`)

**Patient-level folds** (GroupKFold): summary mean/std **ROC-AUC** and **Brier** across folds. Use for stability reporting before locking a single hold-out.

```bash
PYTHONPATH=. python scripts/group_cv_report.py --format longitudinal --data data/demo/ehr_data.csv --model logreg
```

## 3. True external validation (API / UI)

Freeze a trained run, then score a **second CSV** with the same feature contract:

```bash
# API
curl -X POST http://127.0.0.1:8000/v1/jobs/external-validate \
  -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","data_format":"longitudinal","run_id":"<optional_run_id>"}'
```

## 3. External validation job (same feature contract)

```bash
curl -s -X POST localhost:8000/v1/jobs/external-validate -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","data_format":"longitudinal","run_id":"<optional_run_id>"}'
```

Workbench: **Research wizard** (`/research`) or Results → open a run → **External validation** form → poll job → `reports/external_validation_report.json` (also copied into the run trust pack).

Paper curves after retrain: Analytics plots ROC/PR/calibration from `evaluation_report.json`; API `GET /v1/reports/curves`. Full study loop: [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md).

Still required on your side:

- Second **hospital**, **time window**, or **registry** extract in the **same feature contract** (`docs/data_sources_and_schema.md`).
- Freeze **training_manifest** + **evaluation_report** per release; record data SHA-256.
- Document inclusion/exclusion and index/horizon definitions before touching the external file (see [`mimic_lock_checklist.md`](mimic_lock_checklist.md)).

## 4. API / UI visibility

- **`GET /v1/model/metrics`** — reads `reports/evaluation_report.json` and reports whether **`data_sha256` matches** the loaded `model.pkl` manifest.
- **`GET /v1/reports/curves`** — ROC / PR / calibration points (+ bootstrap CIs when present).
- **`GET /v1/reports/methods.md?run_id=`** and **`GET /v1/reports/download.zip?run_id=`** — run-scoped methods note and ZIP including the trust pack.
- Angular **Results** trust checklist + **Home** workspace status surface leakage/SHAP when present.
