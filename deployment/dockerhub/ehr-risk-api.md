# EHR Risk Framework — API image

**Image:** [`ranasl62/ehr-risk-api`](https://hub.docker.com/r/ranasl62/ehr-risk-api)  
**Source:** [github.com/ranasl62/ehr-chronic-disease-risk-prediction](https://github.com/ranasl62/ehr-chronic-disease-risk-prediction)  
**Docs:** [ehr.larucare.com](https://ehr.larucare.com/) · [Docker images page](https://ehr.larucare.com/docker-images)  
**DOI:** [10.5281/zenodo.21448693](https://doi.org/10.5281/zenodo.21448693)

> **Research and education only.** Not a medical device. Not for patient care. Use demo/synthetic data only — never put PHI in public containers.

FastAPI backend for leakage-aware, calibrated chronic-disease–style risk modeling (package `openhealth`, CLI `ehr-ai`). Used by Compose services **api**, **prepare**, and **train**.

## Pull

```bash
docker pull ranasl62/ehr-risk-api:latest
```

Tags: `latest` (default branch), `sha-<short>`, and semver from GitHub `v*` tags.

## Companion web image

```bash
docker pull ranasl62/ehr-risk-web:latest
```

## Quick run (Compose)

Clone the repo (Compose bind-mounts `data/`, `reports/`, `model.pkl`):

```bash
git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
docker compose -f docker-compose.yml -f docker-compose.publish.yml up
```

| Service | URL |
|---------|-----|
| Workbench | http://127.0.0.1:8080 |
| API / OpenAPI | http://127.0.0.1:8000/docs |

Override: `IMAGE_API=ranasl62/ehr-risk-api:latest` in `.env`.

## What this image provides

- REST API: train / compare / predict, health, leakage audit, SHAP, fairness jobs  
- OpenAPI at `/docs`  
- Optional `API_KEY` auth  
- Optional `CORS_ORIGINS` (comma-separated UI origins; empty → localhost :8080/:4200)

```bash
# Example: UI on a separate host talking to this API
docker run --rm -p 8000:8000 \
  -e CORS_ORIGINS=https://ehr-risk-framework-demo.onrender.com \
  ranasl62/ehr-risk-api:latest
```

Compose / `.env`: set `CORS_ORIGINS=` (see `.env.example`). Render/Vercel: set the same runtime env on the API service.

## Links

- Installation: https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/INSTALLATION.md  
- Quickstart: https://ehr.larucare.com/quickstart  
- Limitations: https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/LIMITATIONS.md  
- Support: support@larucare.com  

**License:** MIT
