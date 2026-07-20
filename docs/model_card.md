# Model card — EHR chronic disease risk framework

**Model name:** EHR Chronic Risk Open Framework (reference implementation)  
**Version:** 1.0 (software track)  
**Date:** 2026-07-19  
**Owners:** [`AUTHORS.md`](../AUTHORS.md) — Md Rana Hossain  
**Contact:** [support@larucare.com](mailto:support@larucare.com)  
**License:** MIT ([`LICENSE`](../LICENSE))

## 1. Model details

- **Type:** Binary risk classifier (logistic regression / random forest / XGBoost ± isotonic calibration).
- **Input:** Patient-level multi-window EHR features (`w7d_` / `w30d_` / `w180d_` or single lookback) from longitudinal events truncated at `index_time`.
- **Output:** Probability of incident chronic outcome within horizon \(H\).
- **Artifacts:** `model.pkl`, `reports/evaluation_report.json`, `reports/training_manifest.json`.

## 2. Intended use

- **Primary:** Research and education; reproducible leakage-aware EHR risk pipelines; CDS **prototyping**.
- **Out of scope:** Autonomous diagnosis/treatment; regulated medical device claims; operational deployment without institutional validation.

## 3. Training data

| Track | Source | Notes |
|-------|--------|-------|
| Software verification | `data/raw/paper_synthetic_cohort.csv` | Synthetic; no PHI |
| Teaching demo | `data/raw/ehr_data.csv` | Tiny; CI only |
| Credentialed MIMIC | Local `data/processed/` | Never committed |

## 4. Evaluation

ROC-AUC, PR-AUC, Brier, ECE; patient / temporal splits; leakage audit; optional fairness JSON.  
Synthetic table (local academic package): `research-paper/reports/results_table.csv` (**not** clinical performance).

## 5. Ethical considerations

- No real patient identifiers in the public repository.
- Fairness helpers are exploratory when group columns exist.
- API includes clinical disclaimers (`X-Clinical-Disclaimer`, `/v1/meta`).

## 6. Caveats

- Recalibrate and re-validate on each new site/cohort.
- Prefer `--horizon-days` + explicit `index_time`.
- Full limits: [`LIMITATIONS.md`](../LIMITATIONS.md).

## 7. Citation & feedback

Cite [`CITATION.cff`](../CITATION.cff). Feedback: [`docs/HOW_IT_HELPS.md`](HOW_IT_HELPS.md).
