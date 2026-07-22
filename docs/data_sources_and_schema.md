# Real-world data alignment

This repository uses a **longitudinal event table** (`patient_id`, `timestamp`, optional `icd_code`, vitals/labs, outcome) that mirrors patterns common in U.S. research pipelines. It is **not** a full FHIR or OMOP implementation.

## Canonical CSV schema (longitudinal)

| Column | Role | Notes |
|--------|------|--------|
| `patient_id` | Entity key | Stable per person across rows |
| `timestamp` | Event / visit / claim time | ISO date or datetime; anchor for windows |
| `icd_code` | Diagnosis (optional) | ICD-10-CM style strings in demos |
| `lab_value`, `vital_signs`, `glucose`, `blood_pressure`, `cholesterol`, `age` | Numeric signals | Superset; multi-window code aggregates what exists |
| `label` or `chronic_disease` | Outcome | Patient-level max is used after aggregation |

**Bring your own CSV:** run `python scripts/normalize_longitudinal_csv.py your.csv -o data/processed/normalized.csv` after mapping aliases via `preprocessing/canonical_schema.py`.

## Government and public sources (same *shape*, not same file)

Use only data you are **authorized** to hold. Typical **public, research-grade** U.S. sources that teams join into similar longitudinal tables:

| Source | What it is | How it maps here |
|--------|------------|------------------|
| [CMS DE-SynPUF](https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF) | Synthetic Medicare beneficiaries / claims–like PUF | Beneficiary ID → `patient_id`; claim / service dates → `timestamp`; diagnosis fields → `icd_code` (after cleaning). Labs/vitals usually **not** in claims; add from other tables or leave null. |
| [CMS Limited Data Sets (LDS)](https://www.cms.gov/Research-Statistics-Data-and-Systems/Files-for-Order/LimitedDataSets) | Real Medicare under DUA | Same logical mapping; **restricted** — never commit to a public repo. |
| [HCUP / SID](https://www.hcup-us.ahrq.gov/) | State inpatient databases | Admission date, DX codes → time + `icd_code`; outcomes defined by cohort logic. |
| [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/) | ICU EHR (credentialing) | `subject_id` / `hadm_id` + chart times → `patient_id` + `timestamp`; labs/dx map to feature columns via SQL in `sql/`. |

**NHANES** and survey-style public data differ in structure (wide tables, complex weights); they are valid for population modeling but require a **separate** feature contract than this demo pipeline.

## Production checklist (data + ML)

1. **Cohort definition** — index time, inclusion/exclusion, washout, label horizon (document in protocol).
2. **Leakage** — only pre-index features; patient-level splits for evaluation (`--split-by-patient`).
3. **Imbalance** — XGBoost uses `scale_pos_weight` from training labels; RF uses `class_weight`.
4. **Calibration** — `--calibrate` for probability quality; monitor drift (PSI in `evaluation_report.json` when enabled).
5. **Governance** — BAAs, IRB, minimum necessary; no PHI in logs (see API middleware).

## Synthetic demo files

Bundled teaching fixtures live in **`data/demo/`** (`ehr_data.csv`, `sample_ehr.csv`).  
Paper-scale synthetic cohort: `data/raw/paper_synthetic_cohort.csv`.  
**User imports:** `data/uploads/` (gitignored). Turn demos off in the Datasets UI when working only with your files.

`data/raw/ehr_data.csv` and `data/raw/sample_ehr.csv` are compatibility symlinks into `data/demo/`.

