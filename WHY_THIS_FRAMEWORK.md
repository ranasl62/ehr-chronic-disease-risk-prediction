# Why this framework

Open-source, explainable, **leakage-aware** EHR risk modeling for researchers — not another notebook dump.

**Contact:** [support@larucare.com](mailto:support@larucare.com)

## vs common alternatives

| Need | This framework | Typical notebook / PyHealth-only |
|------|----------------|----------------------------------|
| 5-minute demo | `ehr-ai start` + Angular | Custom env + scripts |
| Temporal leakage | Index/horizon + audit job | Easy to get wrong |
| Calibration | First-class ECE/Brier + isotonic | Often skipped |
| Explainability | SHAP + UI | Optional add-on |
| Config without code | Config Center + task YAML | Edit Python |
| Synthetic + real | Same pipeline + source tags | Separate forks |
| Honesty | `LIMITATIONS.md` in docs + ZIP | Marketing silence |

## Who should use it

- Health-informatics / ML researchers building horizon-based risk models (chronic disease, readmission-style tasks, etc.)
- Teams who need reproducible evaluation packs and leakage audits
- Collaborators demoing CDS-style review on **de-identified / synthetic** data

## Who should not

- Anyone seeking a regulated medical device or live EHR replacement
- Users needing LSTM/Transformer zoo or full OMOP/FHIR servers (see [`LIMITATIONS.md`](LIMITATIONS.md))

## How it helps & feedback

See [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md). Cite via [`CITATION.cff`](CITATION.cff).
