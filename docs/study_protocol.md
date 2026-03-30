# Study protocol template (pre-specification)

Copy this file to `docs/study_protocol_<cohort>.md` and fill before training on real data. **Not IRB approval** — your institution governs human subjects research.

## 1. Objective

- Primary scientific question:
- Model role (screening / risk stratification / research only):

## 2. Data source

- Dataset (e.g. MIMIC-IV v3.x, site-specific warehouse):
- Credentialing / DUA reference:
- Row-level unit (patient, admission, encounter):

## 3. Cohort

- **Inclusion criteria** (bullet list):
- **Exclusion criteria**:
- **Index time** \(t_{\text{index}}\) definition (per patient):
- **Prediction window** for features: \([t_{\text{index}} - W,\, t_{\text{index}})\) with \(W =\) ___ days (or multi-window list).
- **Outcome** definition and **label horizon** \(H\) (only events after \(t_{\text{index}} + \epsilon\)):

## 4. Leakage controls

- Features restricted to pre-index data: yes / no (must be **yes**).
- Train/test split: **patient-level** (required for longitudinal) / other (justify).
- Scripts run: `scripts/leakage_audit.py`, `scripts/validate_training_data.py`.

## 5. Primary and secondary endpoints

- **Primary metric:** e.g. hold-out ROC-AUC, PR-AUC, or clinically anchored metric.
- **Calibration:** target ECE / Brier (report `evaluation_report.json` after train).
- **Fairness:** subgroups and effect sizes (see `scripts/fairness_report.py`).

## 6. Modeling plan

- Algorithms (e.g. XGBoost + logistic baseline):
- **Calibration:** `--calibrate` yes/no (pre-specify).
- **Hyperparameters:** fixed defaults vs nested CV (document).

## 7. Validation plan

- Internal hold-out (this repo): patient-level split, seed ___.
- **External or temporal validation:** describe second split / site / time window (required for strong claims).

## 8. Reproducibility

- Training data file hash: see `reports/training_manifest.json` after `python -m training.train`.
- Git revision recorded in manifest; tag releases for publications.

## 9. Ethics and safety

- IRB / privacy review status:
- Prohibited uses (e.g. no automated treatment without clinician review):
