# Installation

**Maintainer contact:** [support@larucare.com](mailto:support@larucare.com)

## Fastest path — Docker Compose

Works on Windows, macOS, and Linux with Docker Desktop / Engine. Requires **Docker Compose v2.24+** (for publish overlay `!reset`).

### Build from source

```bash
git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose up --build
# equivalent: make researcher-up
# background: make researcher-up-d
```

### Pull published images (no local build)

Images are published to **Docker Hub** by the [`publish-images`](.github/workflows/publish-images.yml) workflow (`main` / `v*` tags). Repos are **Public**.

```bash
docker pull ranasl62/ehr-risk-api:latest
docker pull ranasl62/ehr-risk-web:latest

git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
docker compose -f docker-compose.yml -f docker-compose.publish.yml up
# equivalent: make researcher-up-pull
```

| Image | Docker Hub ref |
|-------|----------------|
| API | `ranasl62/ehr-risk-api:latest` (also `:sha-…`, `:v…`) — [hub](https://hub.docker.com/r/ranasl62/ehr-risk-api) |
| Web | `ranasl62/ehr-risk-web:latest` — [hub](https://hub.docker.com/r/ranasl62/ehr-risk-web) |

Site page: [`docs/docker-images/`](docs/docker-images/).

Pin or override in `.env`: `IMAGE_API=…` · `IMAGE_WEB=…` (see [`.env.example`](.env.example)).

| Service | Host port | Local image | Notes |
|---------|-----------|-------------|--------|
| `web` | **8080** | `ehr-risk-web:local` | Angular + nginx; proxies `/v1` → API |
| `api` | **8000** | `ehr-risk-api:local` | FastAPI; OpenAPI at `/docs` |
| `prepare` | — | same as API | One-shot: trains `model.pkl` if missing |

Open http://127.0.0.1:8080 → Datasets → Train → Results → Predict.

**Persistence:** the repo directory is mounted into `api` / `prepare`, so `data/`, `reports/`, uploads, and `model.pkl` survive `docker compose down`. You still need a clone (or a directory with the same layout) even when pulling images.

**Optional API key:** set `API_KEY` in `.env`; send header `X-API-Key` (UI Config or curl).

**Stop / clean**

```bash
docker compose down              # stop containers (keeps images + host data)
docker compose down --rmi local  # also remove project images
# Host artifacts (optional): rm -f model.pkl; rm -rf reports/runs data/uploads/*
```

### Useful Compose commands

| Goal | Command |
|------|---------|
| Pull published then up | `make researcher-up-pull` |
| Force retrain then up | `FORCE_TRAIN=1 docker compose up --build` |
| Train only | `docker compose --profile train run --rm train` |
| API only | `docker compose up --no-deps api` |
| Custom ports | `UI_PORT=9080 API_PORT=9000 docker compose up --build` |
| Logs | `make researcher-logs` |
| Optional API key / GPU | `cp .env.example .env` and/or `cp docker-compose.override.example.yml docker-compose.override.yml` |

### Publishing images (maintainers)

1. Create a Docker Hub access token with **write** (hub.docker.com → Account Settings → Security).
2. In the GitHub repo: **Settings → Secrets and variables → Actions**, add:
   - `DOCKERHUB_USERNAME` — Docker Hub username (e.g. `ranasl62`)
   - `DOCKERHUB_TOKEN` — the access token (never commit this)
3. Optionally create public repos `ehr-risk-api` and `ehr-risk-web` under that Hub user (or let the first push auto-create them, then set **Public**).
4. Push to `main` or tag `v*` (or run **Publish Docker images** via Actions → workflow_dispatch).
5. Confirm: `docker pull ranasl62/ehr-risk-api:latest` and `docker pull ranasl62/ehr-risk-web:latest`.
6. Confirm each Hub repo **Overview** shows the synced README (`deployment/dockerhub/ehr-risk-api.md` / `ehr-risk-web.md`). The access token needs **Read / Write / Delete** for description sync.

Why two images (not one mega-image)? Keeps the research workbench architecture clear (FastAPI + nginx Angular), faster rebuilds when only UI or only API changes, and matches local `npm start` + `uvicorn` development.

> **Note:** Older docs referenced `ghcr.io/ranasl62/…`. Those GHCR packages were never published (or not public), which is why pulls returned **404**. Use Docker Hub refs above.

---

## Native Python + Angular

Requires **Python 3.10+** and **Node 20+** for the UI.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
ehr-ai init

# terminal 1
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# terminal 2
cd web && npm install && npm start
# http://127.0.0.1:4200 (proxies /v1 to API)
```

### Ubuntu / Debian — `externally-managed-environment`

```bash
sudo apt install -y python3-venv python3-full python3-pip
rm -rf .venv && python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt && pip install -e .
```

Or use [uv](https://docs.astral.sh/uv/): `uv venv .venv && uv pip install -r requirements.txt && uv pip install -e .`

---

## CLI (`ehr-ai`)

```bash
ehr-ai init
ehr-ai doctor
ehr-ai train --task diabetes
ehr-ai compare --task diabetes
ehr-ai evaluate
ehr-ai explain
ehr-ai report
ehr-ai start
```

---

## First train (without Docker prepare)

```bash
PYTHONPATH=. python -m training.train \
  --format longitudinal --data data/demo/ehr_data.csv \
  --model logreg --split-by-patient
```

---

## Environment

Copy [`.env.example`](.env.example) → `.env` (never commit secrets):

- `API_KEY` — optional; when set, send `X-API-Key` (UI Config or curl)
- `UI_PORT` / `API_PORT` — host ports (default 8080 / 8000)
- `IMAGE_API` / `IMAGE_WEB` — image refs (defaults: local tags; Docker Hub with `docker-compose.publish.yml`)
- `CORS_ORIGINS` — browser origins
- `MODEL_PATH` — artifact path (default `model.pkl`)
- `FORCE_TRAIN` — `1` to retrain on every Compose up

Optional Compose merge file: [`docker-compose.override.example.yml`](docker-compose.override.example.yml).

---

## MIMIC-IV

PhysioNet credentialing required for real extracts. Demo/synthetic CSVs need none.  
See [`docs/mimic_lock_checklist.md`](docs/mimic_lock_checklist.md).

---

## Stuck?

Email [support@larucare.com](mailto:support@larucare.com), or open a GitHub issue.  
More: [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md) · [`docs/researcher_quickstart.md`](docs/researcher_quickstart.md).
