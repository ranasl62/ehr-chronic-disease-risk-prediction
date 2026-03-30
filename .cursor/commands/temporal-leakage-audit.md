# Temporal leakage audit

1. **Automated (repo):** run `PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl` or `--data ... --format longitudinal --split-by-patient`. See `docs/mimic_week1_2_runbook.md`.
2. **Index time:** For each cohort, define `t_index`. Every feature must use only events with `t <= t_index` (or strict past-only if policy requires).
3. **Labels:** Outcome only from data **after** `t_index` (or a separate outcome table), never in pre-index feature rows.
4. **Splits:** Longitudinal / MIMIC work → `--split-by-patient`; row-level split risks same-patient leakage.
5. **Multi-window:** Features must be built on history already truncated to the allowed window before aggregation.
6. **Output:** Short checklist in reply; add `tests/` if you find a reproducible bug.
