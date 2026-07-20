# MIMIC Results lock (credentialed)

This repository **cannot** ship MIMIC-IV row data.

**Start here:** [`docs/mimic_lock_checklist.md`](mimic_lock_checklist.md).

After PhysioNet approval:

1. Extract with [`sql/feature_queries.sql`](../sql/feature_queries.sql); normalize to longitudinal CSV with `index_time` + post-index `label` rows. See [`mimic_extract_splits_runbook.md`](mimic_extract_splits_runbook.md).
2. Place extract at `data/processed/mimic_diabetes_cohort.csv` (gitignored).
3. Optional subgroups: `data/processed/mimic_diabetes_groups.csv` with `patient_id,sex,age_band`.
4. Run:

```bash
make -C research-paper mimic-lock
# equivalent:
bash research-paper/scripts/lock_mimic_cohort.sh data/processed/mimic_diabetes_cohort.csv research-paper/reports/mimic
```

Outputs (aggregate only; do not commit PHI):

- `research-paper/reports/mimic/cohort_lock.json` (SHA-256 of extract)
- `experiment_summary.json`, `results_table.csv`
- `leakage_audit.json`, calibration PNGs, SHAP, fairness JSON

## Public software verification (no MIMIC)

```bash
make -C research-paper paper-quick
```

Mark synthetic metrics as **method verification only** — not clinical performance.
