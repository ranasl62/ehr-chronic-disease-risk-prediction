# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] — 2026-07-19

### Framework

- Working package `openhealth` with CLI `ehr-ai` (`start`, `train`, `doctor`, …)
- Task YAML presets under `tasks/` (diabetes, heart failure, custom)
- Dataset health checks, schema map-preview/import, Config Center (`config/workspace.yaml`)
- Named experiment runs under `reports/runs/` with promote-to-active
- Multi-model compare, leakage audit, SHAP, fairness job paths
- Thin OMOP / FHIR subset adapters; clinical-research worklist prototype
- Honest scope docs: `LIMITATIONS.md`, `WHY_THIS_FRAMEWORK.md`

### Product surface

- **Primary UI:** Angular researcher workbench (`web/`) — Streamlit legacy UI removed
- FastAPI researcher/framework routes
- Docker Compose one-command stack (UI :8080, API :8000)
- Downloadable results ZIP and software verification runner (`make paper-quick`)

### Ops / data

- Synthetic software-verification track under `reports/paper/`
- MIMIC lock path: `make mimic-lock` + `docs/mimic_lock_checklist.md` (credentialed extract required; never commit PHI)
- Citation metadata: `CITATION.cff`

### Quality

- Expanded pytest coverage (API, adapters, jobs, schema map, CLI edges)
- CI: train, leakage audit, temporal split, group CV, Docker smoke
