# EHR Risk Framework — Web workbench image

**Image:** [`ranasl62/ehr-risk-web`](https://hub.docker.com/r/ranasl62/ehr-risk-web)  
**Source:** [github.com/ranasl62/ehr-chronic-disease-risk-prediction](https://github.com/ranasl62/ehr-chronic-disease-risk-prediction)  
**Docs:** [ehr.larucare.com](https://ehr.larucare.com/) · [Docker images page](https://ehr.larucare.com/docker-images.html)  
**DOI:** [10.5281/zenodo.21448693](https://doi.org/10.5281/zenodo.21448693)

> **Research and education only.** Not a medical device. Not for patient care. Use demo/synthetic data only — never put PHI in public containers.

Angular research workbench (nginx) for the EHR Risk Framework. Proxies API routes to the companion **api** image.

## Pull

```bash
docker pull ranasl62/ehr-risk-web:latest
```

Tags: `latest` (default branch), `sha-<short>`, and semver from GitHub `v*` tags.

## Companion API image

```bash
docker pull ranasl62/ehr-risk-api:latest
```

## Quick run (Compose)

```bash
git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
docker compose -f docker-compose.yml -f docker-compose.publish.yml up
```

Open **http://127.0.0.1:8080** (UI) and **http://127.0.0.1:8000/docs** (API).

Override: `IMAGE_WEB=ranasl62/ehr-risk-web:latest` in `.env`.

## What this image provides

- Home, Datasets, Train, Results, Analytics, Predict, Config, Docs  
- Proxies `/v1` and related API paths to the backend  

## Links

- Installation: https://github.com/ranasl62/ehr-chronic-disease-risk-prediction/blob/main/INSTALLATION.md  
- Quickstart: https://ehr.larucare.com/quickstart.html  
- Workbench tour: https://ehr.larucare.com/workbench.html  
- Support: support@larucare.com  

**License:** MIT
