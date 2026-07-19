# Runbook: MIMIC-style extract, cohort splits, and leakage checks

This is an **operational checklist** after you receive **PhysioNet MIMIC-IV credentialing**. It does not grant access to MIMIC data.

**Lock path:** [`mimic_lock_checklist.md`](mimic_lock_checklist.md) · [`mimic_results_lock.md`](mimic_results_lock.md) → `make mimic-lock`

## 1. Credentialing and environment

1. Complete [CITI training](https://physionet.org/settings/training/) and sign the MIMIC data use agreement.
2. Provision Postgres with MIMIC-IV (v3.x) or use a hospital mirror approved by your IRB.
3. Confirm table names match your build (`mimiciv_hosp.*`, `mimiciv_icu.*`). Adjust `sql/feature_queries.sql` and `sql/mimic_queries.sql` if your schema differs.

## 2. SQL extraction

1. Start from `sql/feature_queries.sql` (labs, vitals, diagnoses patterns) and `sql/mimic_queries.sql` for cohort-specific joins.
2. **Define explicitly:**
   - **Index time** \(t_{\text{index}}\) per patient (e.g. first qualifying admission end, or cohort entry).
   - **Prediction window** \([t_{\text{index}} - W,\, t_{\text{index}}]\) (or half-open) for features.
   - **Outcome horizon** \(t_{\text{index}} + H\) for labels from post-index diagnoses only.
3. Export event-level rows to CSV or Parquet. Map to the repo longitudinal contract (`patient_id`, `timestamp`, **`index_time`**, optional `icd_code`, numeric labs/vitals, `label`). Use `preprocessing/canonical_schema.py` or `scripts/normalize_longitudinal_csv.py` if column names differ.

## 3. Patient-level split and training

```bash
PYTHONPATH=. python scripts/validate_training_data.py --format longitudinal data/processed/mimic_diabetes_cohort.csv
PYTHONPATH=. python scripts/leakage_audit.py --format longitudinal --data data/processed/mimic_diabetes_cohort.csv \
  --split-by-patient --windows 7,30,180 \
  --horizon-days 365 --index-strategy column --index-time-col index_time

python -m training.train \
  --format longitudinal \
  --data data/processed/mimic_diabetes_cohort.csv \
  --split-by-patient \
  --windows 7,30,180 \
  --horizon-days 365 \
  --index-strategy column \
  --index-time-col index_time \
  --calibrate
```

**Preferred one-shot Results lock:**

```bash
bash scripts/lock_mimic_cohort.sh data/processed/mimic_diabetes_cohort.csv reports/paper/mimic
```

## 4. Leakage audit (artifact)

After training:

```bash
PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl -o reports/leakage_audit.json
```

Expect `patient_disjoint_train_test: true` and `temporal_integrity.passed: true`. Investigate any notes.

## 5. Downstream milestones (same repository)

- **SHAP:** `python scripts/explain_shap.py --artifact model.pkl`
- **Fairness:** `scripts/fairness_report.py` with `patient_id,sex` (or `age_band`) CSV
- **Paper matrix:** `scripts/run_paper_experiments.py`
- **Container smoke:** `bash scripts/docker_smoke.sh`

## 6. Out of scope for this checklist

- IRB, BAAs, or HIPAA minimum-necessary review at your site.
- Cohort-specific ICD/LOINC validation (verify `itemid` / ICD versions for your specific MIMIC build).
- External validation site agreements and reporting (handle under your institutional plan).
