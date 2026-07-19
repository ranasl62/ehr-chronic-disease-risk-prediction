-- =============================================================================
-- MIMIC-IV–style feature extraction (conceptual)
-- Run against a credentialed MIMIC-IV Postgres instance after PhysioNet approval.
-- Table/column names follow MIMIC-IV v3+ conventions; verify against your release.
--
-- Ops checklist: docs/mimic_extract_splits_runbook.md
-- Post-export: preprocessing/canonical_schema.py, scripts/normalize_longitudinal_csv.py
-- QA:       scripts/leakage_audit.py, scripts/validate_training_data.py
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Patient demographics (hosp schema)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, gender, anchor_age, anchor_year, dod
-- FROM mimiciv_hosp.patients;

-- -----------------------------------------------------------------------------
-- Hospital admissions
-- -----------------------------------------------------------------------------
-- SELECT subject_id, hadm_id, admittime, dischtime, admission_type, race
-- FROM mimiciv_hosp.admissions;

-- -----------------------------------------------------------------------------
-- ICU stays (optional for ICU cohorts)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, hadm_id, stay_id, intime, outtime, los
-- FROM mimiciv_icu.icustays;

-- -----------------------------------------------------------------------------
-- Lab events — glucose (itemid 50931 is a common LOINC-mapped glucose in MIMIC)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, hadm_id, charttime, itemid, valuenum AS glucose, valueuom
-- FROM mimiciv_hosp.labevents
-- WHERE itemid = 50931
--   AND valuenum IS NOT NULL
--   AND charttime IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Lab events — HbA1c example (itemid must be verified for your build)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, hadm_id, charttime, itemid, valuenum AS hba1c
-- FROM mimiciv_hosp.labevents
-- WHERE itemid IN (50852)  -- example only — confirm in d_labitems
--   AND valuenum IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Chart events — blood pressure (systolic / diastolic itemids from d_items)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, charttime, itemid, valuenum
-- FROM mimiciv_hosp.chartevents
-- WHERE itemid IN (220050, 220051)  -- verify in mimiciv_hosp.d_items
--   AND valuenum IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Diagnoses — ICD codes (for phenotype / comorbidity flags)
-- -----------------------------------------------------------------------------
-- SELECT subject_id, hadm_id, seq_num, icd_code, icd_version
-- FROM mimiciv_hosp.diagnoses_icd;

-- -----------------------------------------------------------------------------
-- Derived cohort: incident diabetes (align with docs/study_protocol_mimic_diabetes.md)
-- -----------------------------------------------------------------------------
-- 1) Choose index_time per subject (e.g., first qualifying discharge).
-- 2) Exclude prevalent diabetes (ICD / labs) on or before index_time.
-- 3) Restrict labs/vitals to [index_time - W, index_time] (or half-open upper bound).
-- 4) Set label=1 iff first diabetes evidence in (index_time, index_time + H]; else 0.
-- 5) Export event-level CSV with columns:
--      patient_id, timestamp, index_time, glucose, blood_pressure, cholesterol,
--      age, sex (optional), icd_code, label
-- 6) Lock Results: bash scripts/lock_mimic_cohort.sh data/processed/mimic_diabetes_cohort.csv
--
-- Pseudo-SQL — implement joins in your MIMIC Postgres client; do not commit extracts.
