# Installation

**Maintainer contact:** [support@larucare.com](mailto:support@larucare.com)

## Fastest path — Docker Compose

Works on Windows, macOS, and Linux with Docker Desktop / Engine:

```bash
git clone https://github.com/ranasl62/ehr-chronic-disease-risk-prediction.git
cd ehr-chronic-disease-risk-prediction
docker compose up --build
# equivalent: make researcher-up
# or after pip install -e .:  ehr-ai start
```

| Service | URL |
|---------|-----|
| Angular workbench | http://127.0.0.1:8080 |
| FastAPI / OpenAPI | http://127.0.0.1:8000/docs |

`prepare` trains `model.pkl` only if missing (or set `FORCE_TRAIN=1`). Requires **Docker Compose v2.20+**.

### Useful Compose commands

| Goal | Command |
|------|---------|
| Force retrain then up | `FORCE_TRAIN=1 docker compose up --build` |
| Train only | `docker compose --profile train run --rm train` |
| API only | `docker compose up --no-deps api` |

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
  --format longitudinal --data data/raw/ehr_data.csv \
  --model logreg --split-by-patient
```

---

## Environment

Copy [`.env.example`](.env.example) → `.env` (never commit secrets):

- `API_KEY` — optional; required for non-local exposure
- `CORS_ORIGINS` — browser origins
- `MODEL_PATH` — artifact path (default `model.pkl`)

---

## MIMIC-IV

PhysioNet credentialing required for real extracts. Demo/synthetic CSVs need none.  
See [`docs/mimic_lock_checklist.md`](docs/mimic_lock_checklist.md).

---

## Stuck?

Email [support@larucare.com](mailto:support@larucare.com), or open a GitHub issue.  
More: [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md) · [`docs/researcher_quickstart.md`](docs/researcher_quickstart.md).
