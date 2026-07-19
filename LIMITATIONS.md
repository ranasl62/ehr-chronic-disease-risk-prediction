# Limitations

This framework is an **open research / clinical decision-support prototyping** toolkit.
It is **not** a regulated medical device.

**Questions or feedback:** [support@larucare.com](mailto:support@larucare.com) · [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md)

## A. Clinical / legal / safety

- Not for diagnosis, treatment, or triage without institutional governance
- No FDA/CE claim; outputs are research probabilities only
- User is responsible for PHI, IRB, DUA, and PhysioNet rules
- Local API is unsafe on public networks without `API_KEY`
- Clinical-research mode is a **prototype**, not production CDS

## B. Data coverage

- Canonical path: tabular/longitudinal events → window aggregates
- Not full free-text NLP, imaging, waveforms, or genomics
- FHIR/OMOP adapters are **subset** importers, not full CDM/FHIR servers
- MIMIC requires **user credentials**; restricted data is never shipped
- Schema mapping cannot fix wrong labels or upstream leakage
- Default upload size limit ~50MB
- SQL import is read-only `SELECT` / `WITH … SELECT`

## C. Modeling / ML

- Supported: logistic regression, random forest, XGBoost, optional LightGBM
- **LSTM / Transformers / foundation models: not supported**
- Multi-model compare ranks by hold-out ROC-AUC — not full AutoML/HPO
- Tiny cohorts → unstable or undefined AUC (framework warns)
- Calibration improves probability quality; does not guarantee clinical utility
- SHAP explains the fitted model — not causal effects

## D. Evaluation / science

- Default splits are research hold-outs; multi-site validation is user-provided
- Fairness requires group columns; otherwise skipped with an explicit reason
- Leakage audit catches common classes — not every contamination pattern
- Synthetic results are for method/CI verification — not clinical performance claims

## E. Platform / ops

- Single-host research workbench; not multi-tenant SaaS
- One heavy job at a time (not a cluster scheduler)
- Filesystem experiment runs — not a full MLflow/W&B replacement
- No Model Hub / community marketplace in this release
- Windows/macOS ease depends on Docker Desktop
- User backs up `reports/runs/` and config

## F. Customization boundaries

- Configurable: tasks, windows, horizon, splits, models, calibrate, persona
- Formal plugin marketplace not included
- Advanced feature engineering beyond YAML may require Python changes

## G. Forbidden implications

We do **not** claim to:

- Solve all healthcare AI
- Replace hospital EHR systems
- Be state-of-the-art on every MIMIC benchmark
- Be “bias-free” or to have “solved fairness”
