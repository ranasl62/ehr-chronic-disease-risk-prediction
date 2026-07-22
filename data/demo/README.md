# Bundled demo datasets

Teaching / CI fixtures only — **not** clinical data and not for patient care.

| File | Role |
|------|------|
| `ehr_data.csv` | Tiny longitudinal demo (~10 patients) for Run demo / smoke tests |
| `sample_ehr.csv` | Legacy tabular sample |

**Your real data** goes in `data/uploads/` via the Datasets page (never commit PHI).

In the workbench: **Datasets → Browse → “Show bundled demo datasets”** can be turned off so only your imports appear.

Paper-scale synthetic cohort remains under `data/raw/paper_synthetic_cohort.csv` (also tagged as bundled synthetic in the catalog).

Compatibility: `data/raw/ehr_data.csv` and `data/raw/sample_ehr.csv` are symlinks into this folder so older scripts keep working.
