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
Hub: [ehr-risk-api](https://hub.docker.com/r/ranasl62/ehr-risk-api) · [ehr-risk-web](https://hub.docker.com/r/ranasl62/ehr-risk-web) · Docs: [`docker-images.html`](docker-images.html)  
Override: `IMAGE_API` / `IMAGE_WEB` in `.env`. Ports **8080** (UI) / **8000** (API). Optional `API_KEY`.

Optional: `cp .env.example .env` · GPU/overrides: `docker-compose.override.example.yml`.

| | |
|--|--|
| Angular workbench | http://127.0.0.1:8080 (`ehr-risk-web:local` or Docker Hub web image) |
| FastAPI + OpenAPI | http://127.0.0.1:8000/docs (`ehr-risk-api:local` or Docker Hub api image) |
| Results ZIP | Results page, or `GET /v1/reports/download.zip` |

`data/`, `reports/`, and `model.pkl` persist on the host (bind mount). Research / education only — contact [support@larucare.com](mailto:support@larucare.com).

### In the Angular UI

1. **Home** — checklist / API healthy  
2. **Datasets** — demo or import; run **Dataset health**  
3. **Train** — task preset or manual → Start / Compare  
4. **Analytics** — charts + filterable tables  
5. **Results** — metrics, figures, ZIP  
6. **Predict** — schema form  
7. **Config** — workspace + UI preferences  

CLI: `pip install -e .` then `ehr-ai start` / `ehr-ai train --task diabetes` — [`INSTALLATION.md`](../INSTALLATION.md).

## Option B — Local API + Angular

```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
cd web && npm install && npm start
```

## Bring your own CSV

1. Match the longitudinal contract (`patient_id`, `timestamp`, labs/vitals, `label`).  
2. Optional: `index_time` + horizon via UI or CLI.  
3. [`DATA_GUIDE.md`](../DATA_GUIDE.md) · [`docs/data_sources_and_schema.md`](data_sources_and_schema.md).  

## Next

- Architecture: [`ARCHITECTURE.md`](../ARCHITECTURE.md)  
- Verification matrix: `make paper-quick`  
- MIMIC lock: [`mimic_lock_checklist.md`](mimic_lock_checklist.md)  
- Cite: [`CITATION.cff`](../CITATION.cff)
