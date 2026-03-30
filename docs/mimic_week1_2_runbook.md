# Weeks 1–2 runbook: MIMIC-IV extract, splits, leakage checks

This is an **operational checklist** after you receive **PhysioNet MIMIC-IV credentialing**. It does not grant access to MIMIC data.

## 1. Credentialing and environment

1. Complete [CITI training](https://physionet.org/settings/training/) and sign the MIMIC data use agreement.
2. Provision Postgres with MIMIC-IV (v3.x) or use a hospital mirror approved by your IRB.
3. Confirm table names match your build (`mimiciv_hosp.*`, `mimiciv_icu.*`). Adjust `sql/feature_queries.sql` and `sql/mimic_queries.sql` if your schema differs.

## 2. SQL extraction

1. Start from `sql/feature_queries.sql` (labs, vitals, diagnoses patterns) and `sql/mimic_queries.sql` for cohort-specific joins.
2. **Define explicitly:**
   - **Index time** \(t_{\text{index}}\) per patient (e.g. first qualifying admission end, or cohort entry).
   - **Prediction window** \([t_{\text{index}} - W,\, t_{\text{index}})\) for features.
   - **Outcome horizon** \(t_{\text{index}} + H\) for labels from post-index diagnoses only.
3. Export event-level rows to CSV or Parquet. Map to the repo longitudinal contract (`patient_id` / `subject_id`, `timestamp` / `charttime`, optional `icd_code`, numeric labs/vitals, `label`). Use `preprocessing/canonical_schema.py` or `scripts/normalize_longitudinal_csv.py` if column names differ.

## 3. Patient-level split and training

```bash
PYTHONPATH=. python scripts/validate_training_data.py --format longitudinal data/processed/mimic_cohort.csv
PYTHONPATH=. python scripts/leakage_audit.py --format longitudinal --data data/processed/mimic_cohort.csv --split-by-patient --windows 7,30,180

python -m training.train \
  --format longitudinal \
  --data data/processed/mimic_cohort.csv \
  --split-by-patient \
  --windows 7,30,180 \
  --calibrate
```

## 4. Leakage audit (artifact)

After training:

```bash
PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl -o reports/leakage_audit.json
```

Expect `patient_disjoint_train_test: true` when `split_method` is `patient_group`. Investigate any notes.

## 5. Weeks 3–4 hooks (same repo)

- **SHAP (holdout aligned with training):** `python scripts/explain_shap.py --artifact model.pkl`
- **Fairness slices:** build a `patient_id,subgroup` CSV for the cohort and run `scripts/fairness_report.py` (see script help).
- **Docker smoke:** `bash scripts/docker_smoke.sh`

## 6. What this runbook does *not* do

- IRB, BAAs, or HIPAA minimum-necessary review.
- Cohort-specific ICD/LOINC validation (you must verify `itemid` / ICD versions for your MIMIC build).
- External validation (Week 5+ paper work).
