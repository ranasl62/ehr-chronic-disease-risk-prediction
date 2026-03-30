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
PYTHONPATH=. python scripts/group_cv_report.py --format longitudinal --data data/raw/ehr_data.csv --model logreg
```

## 3. True external validation (your action items)

- Second **hospital**, **time window**, or **registry** extract in the **same feature contract** (`docs/data_sources_and_schema.md`).
- Freeze **training_manifest** + **evaluation_report** per release; cite SHA-256 in the paper appendix.
- Prefer **pre-spec** in `docs/study_protocol.md` before touching the external file.

## 4. API / UI visibility

- **`GET /v1/model/metrics`** — reads `reports/evaluation_report.json` and reports whether **`data_sha256` matches** the loaded `model.pkl` manifest.
- **Streamlit** sidebar shows the same alignment flag and headline metrics when the JSON exists.
