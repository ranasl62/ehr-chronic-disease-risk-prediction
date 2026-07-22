# MIMIC Results lock checklist

Credentialed MIMIC-IV data **never** goes in git. This checklist ends with a local `cohort_lock.json` (SHA-256 only).

Cross-links: [access + outreach + test plan](mimic_access_and_outreach.md) · [extract runbook](mimic_extract_splits_runbook.md) · [lock details](mimic_results_lock.md) · [SQL templates](../sql/feature_queries.sql) · [data schema](data_sources_and_schema.md)

## 1. Credentials

- [ ] PhysioNet CITI + signed MIMIC-IV DUA
- [ ] Local Postgres (or warehouse) with `mimiciv_hosp` (or equivalent) loaded
- [ ] Institutional rules for PHI / restricted data documented **outside** this repo

## 2. Extract

- [ ] Define inclusion/exclusion and index time for your study (document locally)
- [ ] Run / adapt queries in [`sql/feature_queries.sql`](../sql/feature_queries.sql)
- [ ] Produce a longitudinal table with at least:
  - `patient_id`, `timestamp`, `index_time`
  - clinical numeric columns as available
  - `label` (incident outcome in post-index horizon)
- [ ] Normalize column names if needed:

```bash
PYTHONPATH=. python scripts/normalize_longitudinal_csv.py YOUR_EXTRACT.csv \
  -o data/processed/mimic_diabetes_cohort.csv
```

## 3. Optional subgroups (fairness)

- [ ] `data/processed/mimic_diabetes_groups.csv` with `patient_id,sex,age_band`

## 4. Validate

```bash
PYTHONPATH=. python scripts/validate_training_data.py \
  --format longitudinal data/processed/mimic_diabetes_cohort.csv
```

## 5. Lock

```bash
make -C research-paper mimic-lock
# or:
bash research-paper/scripts/lock_mimic_cohort.sh \
  data/processed/mimic_diabetes_cohort.csv \
  research-paper/reports/mimic
```

| Artifact | Safe to share publicly? |
|----------|-------------------------|
| `cohort_lock.json` (`data_sha256`, `locked_at_utc`) | Yes — SHA only |
| Aggregate metrics / figures under `research-paper/reports/mimic/` | Yes — no row-level data |
| Extract CSV | **Never** commit |

## Stop conditions

- Missing PhysioNet approval → stay on the public synthetic verification track (`make -C research-paper paper-quick`)
- Lock script exits 1 on missing CSV → complete steps 2–4 first
- Never push `data/processed/mimic_*.csv` to GitHub
