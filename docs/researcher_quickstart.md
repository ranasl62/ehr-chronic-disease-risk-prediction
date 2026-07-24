# Researcher quickstart (≈15 minutes)

**Audience:** health-informatics / ML researchers.  
**Not** a clinical product — research decision-support prototyping only.

**Help / feedback:** [support@larucare.com](mailto:support@larucare.com) · [`docs/HOW_IT_HELPS.md`](HOW_IT_HELPS.md)

## Option A — Docker (recommended)

**Build from source**

```bash
git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose up --build
# or: make researcher-up
# stop: docker compose down
```

**Pull published images** (Docker Hub; public)

```bash
docker pull ranasl62/ehr-risk-api:latest
docker pull ranasl62/ehr-risk-web:latest

git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
docker compose -f docker-compose.yml -f docker-compose.publish.yml up
# or: make researcher-up-pull
```

Defaults: `ranasl62/ehr-risk-api:latest` · `ranasl62/ehr-risk-web:latest`  
Hub: [ehr-risk-api](https://hub.docker.com/r/ranasl62/ehr-risk-api) · [ehr-risk-web](https://hub.docker.com/r/ranasl62/ehr-risk-web) · Docs: [`docker-images`](docker-images/)  
Override: `IMAGE_API` / `IMAGE_WEB` in `.env`. Ports **8080** (UI) / **8000** (API). Optional `API_KEY`.

Optional: `cp .env.example .env` · GPU/overrides: `docker-compose.override.example.yml`.

| | |
|--|--|
| Angular workbench | http://127.0.0.1:8080 (`ehr-risk-web:local` or Docker Hub web image) |
| FastAPI + OpenAPI | http://127.0.0.1:8000/docs (`ehr-risk-api:local` or Docker Hub api image) |
| Results ZIP | Results page, or `GET /v1/reports/download.zip` |

`data/`, `reports/`, and `model.pkl` persist on the host (bind mount). For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future. Contact [support@larucare.com](mailto:support@larucare.com).

### In the Angular UI

1. **Home** — checklist / API healthy; **Start research wizard** for a guided study  
2. **Research** (`/research`) — health → train → trust → leakage → external → export  
3. **Datasets** — demo or import; run **Dataset health**  
4. **Train** — task preset or manual → Start / Compare  
5. **Analytics** — cohort charts + ROC/PR/calibration (after retrain); PNG / print  
6. **Results** — metrics, trust, SHAP, external val, ZIP  
7. **Predict** — schema form; export session JSON  
8. **Config** — workspace + UI preferences  

Full loop: [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md).

CLI: `pip install -e .` then `ehr-ai start` / `ehr-ai train --task diabetes` — [`INSTALLATION.md`](../INSTALLATION.md).

## Option B — Local API + Angular

```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
cd web && npm install && npm start
```

## Bring your own CSV

1. Match the longitudinal contract (`patient_id`, `timestamp`, labs/vitals, `label`).  
2. In the UI: **Datasets → File upload** (files land in `data/uploads/`).  
3. Optionally uncheck **Show bundled demo datasets** so only your imports appear (`GET /v1/datasets?include_demo=false`).  
4. Bundled teaching fixtures live in [`data/demo/`](../data/demo/) — not for clinical claims.  
5. Optional: `index_time` + horizon via UI or CLI. Tiny `data/demo/ehr_data.csv` has no `index_time` — use `custom` / `last_event`, or `data/raw/paper_synthetic_cohort.csv` for diabetes / horizon / readmission tasks.  
6. [`docs/data_sources_and_schema.md`](data_sources_and_schema.md) · MIMIC: [`mimic_access_and_outreach.md`](mimic_access_and_outreach.md).  

## Next

- Research workflow: [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) · gap list: [`GAP_CLOSURES.md`](GAP_CLOSURES.md)  
- Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md)  
- Verification matrix (local academic package): `make -C research-paper paper-quick`  
- MIMIC lock: [`mimic_lock_checklist.md`](mimic_lock_checklist.md)  
- Cite: [`CITATION.cff`](../CITATION.cff)
- Thresholds / operating points: `GET /v1/reports/thresholds` (optional on-disk `reports/threshold_operating_points.json`)
- Trust pack: per-run `reports/runs/<id>/trust_pack.json` (not a `/v1/reports/trust-pack` endpoint)
