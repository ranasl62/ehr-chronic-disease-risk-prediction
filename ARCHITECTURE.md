# Architecture

Open-source **leakage-aware EHR risk framework**  
Package: `openhealth` · CLI: `ehr-ai` · UI: Angular workbench · API: FastAPI  

Research / education prototype — **not** a medical device.

**Maintainer:** Md Rana Hossain · [support@larucare.com](mailto:support@larucare.com)

---

## 1. Goals

| Goal | How the architecture supports it |
|------|----------------------------------|
| Temporal integrity | Features truncated at `index_time`; labels only in post-index horizon |
| Reproducible experiments | Task YAML, named runs under `reports/runs/`, manifests with data SHA-256 |
| Trust surfaces | Dataset health, leakage audit, calibration (ECE/Brier), SHAP, fairness helpers |
| Fast researcher loop | One Docker stack: import → health → train/compare → analytics → predict |
| Honest scope | Explicit non-goals in [`LIMITATIONS.md`](LIMITATIONS.md) |

---

## 2. System context

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Researcher / developer                        │
│         browser (:8080)  ·  CLI (ehr-ai)  ·  OpenAPI (:8000)     │
└───────────────┬─────────────────────┬───────────────┬───────────┘
                │                     │               │
                ▼                     ▼               ▼
        ┌───────────────┐    ┌──────────────┐  ┌─────────────┐
        │ Angular web/  │    │ openhealth   │  │ scripts/    │
        │ workbench     │───▶│ CLI + facade │  │ audits/SHAP │
        └───────┬───────┘    └──────┬───────┘  └──────┬──────┘
                │                   │                  │
                └─────────┬─────────┴──────────────────┘
                          ▼
                 ┌────────────────────┐
                 │ FastAPI api/       │
                 │ jobs · datasets ·  │
                 │ predict · config   │
                 └─────────┬──────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │ Core ML stack                             │
        │ feature_engineering/ · training/ · models/│
        │ explainability/ · fairness/               │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │ Artifacts (host filesystem)               │
        │ model.pkl · reports/ · config/workspace.yaml │
        │ data/raw (demo) · data/uploads (gitignored)  │
        └──────────────────────────────────────────┘
```

---

## 3. Runtime topology (Docker)

```text
docker compose up --build
        │
        ├─ prepare  → train model.pkl if missing (scripts/docker_prepare.sh)
        ├─ api      → :8000  (FastAPI, mounts repo)
        └─ web      → :8080  (Angular + nginx proxy to api)

Optional: --profile train → one-shot training container
```

Local alternative: `uvicorn api.main:app` + `cd web && npm start` (proxy to API).

---

## 4. End-to-end data / ML pipeline

```text
1. Ingest
   CSV / TSV / JSON / XLSX / form / SQL SELECT / OMOP·FHIR subset
        → normalize to longitudinal or tabular contract
        → data/uploads/*.csv (gitignored)

2. Health
   openhealth/health.py · GET /v1/datasets/health
        → blockers / warnings (missing id, tiny N, horizon without index_time)

3. Task
   tasks/*.yaml · Config Center
        → horizon, windows [7,30,180], split, model_kind, calibrate

4. Features
   feature_engineering/ (index truncate + multi-window aggregates)
        → columns w7d_* / w30d_* / w180d_*

5. Train / compare
   training/train.py · openhealth/compare.py
        → patient or temporal split · optional isotonic calibration
        → reports/runs/<id>/ · optional promote to active model.pkl

6. Trust jobs
   leakage_audit · SHAP · fairness
        → JSON / PNG under reports/

7. Serve
   POST /v1/predict · /explain · Angular Predict
        → risk probability + optional SHAP explanation
```

**Hard rule:** feature events must satisfy \(t \le t_{\text{index}}\). Labels use only events in \((t_{\text{index}},\, t_{\text{index}}+H]\).

---

## 5. Component map

| Layer | Responsibility | Primary paths |
|-------|----------------|---------------|
| **Workbench UI** | Datasets, Train, Results, Analytics, Config, Predict, Docs | `web/src/app/` |
| **HTTP API** | Auth gate, jobs, datasets, framework routes, predict | `api/main.py`, `api/researcher_routes.py`, `api/framework_routes.py`, `api/jobs.py` |
| **Framework package** | Tasks, health, compare, config, events, adapters, CLI | `openhealth/` |
| **Feature / cohort** | Index/horizon integrity, multi-window tables | `feature_engineering/` |
| **Training** | Splits, train CLI, manifests | `training/` |
| **Models** | LogReg, RF, XGBoost, calibration | `models/` |
| **Explain / fairness** | SHAP, subgroup reports | `explainability/`, `fairness/`, `scripts/` |
| **Config** | Workspace YAML | `config/workspace.yaml` |
| **Tasks** | Presets (diabetes, heart_failure, custom) | `tasks/*.yaml` |
| **SQL templates** | MIMIC-style extract patterns | `sql/` |

---

## 6. Job model

Heavy work runs as **one job at a time** (research workbench, not a cluster):

| Job | Endpoint (summary) | Output |
|-----|-------------------|--------|
| Train | `POST /v1/jobs/train` | `model.pkl` or named run |
| Compare | `POST /v1/jobs/compare` | ranking + optional promote |
| Leakage audit | `POST /v1/jobs/leakage-audit` | `reports/leakage_audit.json` |
| SHAP | `POST /v1/jobs/shap` | summary figure / JSON |
| Fairness | fairness job route | subgroup JSON |

Poll `GET /v1/jobs/{id}`; cancel when supported. Progress appears in the UI.

---

## 7. Configuration surfaces

| Surface | What it controls |
|---------|------------------|
| `tasks/*.yaml` | Target, horizon, windows, suggested data path, default model |
| `config/workspace.yaml` | Persona, train defaults, compare models, UI prefs |
| Angular Config Center | Same workspace settings + theme/density |
| Env (`.env`) | `API_KEY`, CORS, `MODEL_PATH`, rate limits |

---

## 8. Trust & governance hooks

- Dataset **health** before train continue (blockers vs warnings)
- **Leakage audit** (patient-disjoint + temporal integrity)
- **Calibration** metrics (Brier, ECE) and optional isotonic
- **SHAP** local explanations via API / UI
- Response header / `/v1/meta` clinical disclaimer
- Optional `API_KEY` for non-local exposure

---

## 9. Explicit non-goals

See [`LIMITATIONS.md`](LIMITATIONS.md). Not in this architecture:

- Regulated medical device / multi-tenant PHI SaaS
- LSTM / Transformers / foundation-model training
- Full OMOP CDM or FHIR server (subset importers only)
- Model Hub / plugin marketplace
- Cluster job scheduler / full MLflow replacement

---

## 10. Related docs

| Doc | Purpose |
|-----|---------|
| [`INSTALLATION.md`](INSTALLATION.md) | Docker & native install |
| [`DATA_GUIDE.md`](DATA_GUIDE.md) | Schema & ingest |
| [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md) | Benefits & feedback |
| [`docs/researcher_quickstart.md`](docs/researcher_quickstart.md) | 15-minute tour |
| [`WHY_THIS_FRAMEWORK.md`](WHY_THIS_FRAMEWORK.md) | Positioning |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Honest boundaries |
