# Early Chronic Disease Risk Prediction from Electronic Health Records: An End-to-End Clinical Intelligence System

**Authors:** Md Rana Hossain  
**Affiliation:** Maharishi International University ([mdrana.hossain@miu.edu](mailto:mdrana.hossain@miu.edu), [LinkedIn](https://www.linkedin.com/in/mdranahossain/))  
**Manuscript type:** Full paper (IEEE-style Markdown draft — convert to LaTeX IEEEtran for submission)

---

## Abstract

**Background:** Chronic diseases account for a disproportionate share of morbidity and healthcare expenditure. Electronic health records (EHRs) encode longitudinal signals that may support earlier risk stratification when modeled with rigorous temporal discipline.

**Methods:** We present an end-to-end **clinical intelligence pipeline** that (1) extracts MIMIC-IV–style longitudinal tables via SQL templates, (2) constructs **multi-scale temporal features** (7-, 30-, and 180-day lookbacks) anchored at a defined index time, (3) trains **gradient-boosted trees (XGBoost)** and baselines (logistic regression, Random Forest, optional LightGBM) with **isotonic probability calibration**, (4) evaluates **discrimination, calibration, and lead-time–aware metrics**, and (5) exposes **SHAP-based explanations** through a **FastAPI** service and **Streamlit** dashboard.

**Results:** On a public-style demo cohort (software verification only), the system reproduces a full train→evaluate→explain→deploy loop. Empirical performance on institutional data is reported as *TBD* pending IRB-approved cohorts.

**Conclusions:** The architecture demonstrates **reproducible**, **interpretable**, and **deployment-ready** chronic disease risk analytics suitable for prospective validation and clinical decision-support *research*.

**Index Terms** — Electronic health records, chronic disease, gradient boosting, calibration, SHAP, clinical decision support.

---

## I. Introduction

Chronic conditions such as diabetes mellitus, hypertension, and cardiovascular disease progress over years. EHRs capture labs, vitals, diagnoses, and encounters that may precede clinical decompensation. Machine learning on such data must respect **temporal ordering** and **avoid information leakage** from the future. This work contributes a **reference implementation** with explicit data lineage, multi-window aggregation, calibration, explainability hooks, and containerized deployment.

---

## II. Related Work

Supervised learning on EHR cohorts is extensive; key challenges include **label definition**, **missing data**, **dataset shift**, and **miscalibrated probabilities**. Post-hoc calibration (isotonic / Platt) and **SHAP** explanations are common in healthcare ML. Our emphasis is **systems integration**: SQL → Python ETL → training artifacts → JSON API + UI.

---

## III. Methodology

### A. Data and cohort

We target **MIMIC-IV**-compatible schemas (`sql/mimic_queries.sql`). A longitudinal **demo extract** (`data/raw/ehr_data.csv`) supports CI and UI smoke tests. Production use requires **PhysioNet credentialing** and institutional approvals. **U.S. public PUF-style** sources (e.g., CMS DE-SynPUF, HCUP) can be normalized to the same longitudinal column contract; see `docs/data_sources_and_schema.md`.

### B. Temporal feature engineering

For each patient, let \(t^\*\) denote the **anchor time** (last observed event in the demo; in studies, a prespecified prediction time). For window length \(W \in \{7, 30, 180\}\) days, features aggregate events with \(t \in [t^\* - W, t^\*]\). Columns are prefixed `w7d_`, `w30d_`, `w180d_` to preserve scale-specific physiology. **Leakage control:** truncate all rows to \(t < t^\*\) if the label uses post-\(t^\*\) evidence.

### C. Models

- **Logistic regression** (scaled linear baseline).  
- **Random Forest.**  
- **XGBoost** (primary).  
- **LightGBM** (optional).  
- **Calibration:** `CalibratedClassifierCV` with isotonic mapping and internal CV.

### D. Evaluation

**ROC-AUC**, **PR-AUC**, **Brier score**, precision/recall/F1 at a fixed threshold, and **reliability diagrams**. **Lead-time** summaries compare prediction time to diagnosis time when timestamps are aligned (`compute_lead_time_days`, `print_lead_time_summary`).

### E. Explainability

**SHAP TreeExplainer** (XGBoost, LightGBM, RF) and **LinearExplainer** for logistic pipelines. **Calibrated** ensembles unwrap to the underlying base estimator for attribution. Explanations are serialized to JSON in the API (`/v1/predict`, `/explain`).

### F. Deployment

**Docker** (`deployment/Dockerfile`), **FastAPI** inference, **Streamlit** dashboard, and **GitHub Actions** CI for import/unit smoke tests.

---

## IV. Experiments

### A. Setup

Train/test split with stratification when feasible; **patient-level** splitting is required for production studies (extend split utility). Random seed logged in `evaluation_report.json`.

### B. Metrics

Report discrimination and calibration on held-out data; include **calibration curves** (`reports/calibration_holdout.png`).

### C. Ablations

Planned: single-window vs multi-window, uncalibrated vs calibrated, XGBoost vs LightGBM.

---

## V. Results

*Institutional results pending.* The repository ships **deterministic** training outputs: `model.pkl`, `reports/evaluation_report.json`, `reports/feature_importance.json`.

---

## VI. Discussion

**Strengths:** explicit temporal windows, calibration, SHAP JSON, API/UI, Docker, CI.  
**Limitations:** demo cohort size; lack of full MIMIC-IV evaluation in-repo; calibration with small \(n\) is unstable; fairness analysis is stubbed.

---

## VII. Conclusion

We present a **production-shaped** EHR clinical intelligence stack for chronic disease risk with **research-grade** evaluation and **interpretability**. Prospective validation, fairness auditing, and regulatory pathways are required before clinical deployment.

---

## Acknowledgments

MIMIC-IV is a licensed resource. This draft supports portfolio and independent research documentation and does not constitute medical or legal advice.

## References

[1] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” KDD, 2016.  
[2] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” NeurIPS, 2017.  
[3] A. E. W. Johnson et al., “MIMIC-IV,” PhysioNet, 2023.  
[4] A. Niculescu-Mizil and R. Caruana, “Predicting good probabilities with supervised learning,” ICML, 2005.

---

## Appendix A — System components (implementation map)

| Agent (concept) | Repository anchor |
|-----------------|-------------------|
| Data engineering | `sql/mimic_queries.sql`, `preprocessing/feature_engineering.py`, `preprocessing/mimic_pipeline.py` |
| Machine learning | `training/train.py`, `models/train.py`, `models/calibration.py` |
| XAI | `explainability/shap_explainer.py`, `explainability/explanation.py` |
| API | `api/main.py` |
| UI | `dashboard/app.py` |
| MLOps | `deployment/Dockerfile`, `.github/workflows/ci.yml` |
