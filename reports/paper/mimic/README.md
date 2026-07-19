Credentialed MIMIC-IV extracts are **not** stored in this repository.

**Checklist:** [`docs/mimic_lock_checklist.md`](../../docs/mimic_lock_checklist.md)

When PhysioNet access is ready:

```bash
bash scripts/lock_mimic_cohort.sh data/processed/mimic_diabetes_cohort.csv reports/paper/mimic
```

Expected outputs here after lock:

| File | Purpose |
|------|---------|
| `cohort_lock.json` | SHA-256 of extract (no PHI) |
| `experiment_summary.json` / `results_table.csv` | Metrics |
| `leakage_audit.json` | Temporal + patient-disjoint checks |
| `shap_summary.png` | Holdout SHAP |
| `fairness_*.json` | Subgroup metrics |

Until then, use the **public software verification** track under `reports/paper/` — label those numbers as method verification, not clinical performance.

See [`docs/mimic_results_lock.md`](../../docs/mimic_results_lock.md).
