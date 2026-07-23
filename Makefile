.PHONY: install test test-web test-all train train-patient train-temporal leak-audit shap docker-smoke cv-report researcher-up researcher-up-d researcher-up-pull researcher-up-pull-d researcher-down researcher-logs

# Prefer project venv when present (.venv2 is the working env on this machine).
PYTHON ?= $(shell if [ -x .venv2/bin/python ]; then echo .venv2/bin/python; elif [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

install:
	$(PYTHON) -m pip install -r requirements.txt && $(PYTHON) -m pip install -e .

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -m "not e2e" -q --tb=short

test-cov:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -m "not e2e" -q --tb=short

test-web:
	cd web && npm run test:ci

test-all: test test-web

train:
	PYTHONPATH=. python -m training.train --format longitudinal --data data/raw/ehr_data.csv --model logreg

train-patient:
	PYTHONPATH=. python -m training.train --format longitudinal --data data/raw/ehr_data.csv --model logreg --split-by-patient --bootstrap-samples 300

train-temporal:
	PYTHONPATH=. python -m training.train --format longitudinal --data data/raw/ehr_data.csv --model logreg --temporal-split

cv-report:
	PYTHONPATH=. python scripts/group_cv_report.py --format longitudinal --data data/raw/ehr_data.csv --model logreg

leak-audit:
	PYTHONPATH=. python scripts/leakage_audit.py --artifact model.pkl

shap:
	PYTHONPATH=. python scripts/explain_shap.py --artifact model.pkl

docker-smoke:
	bash scripts/docker_smoke.sh

researcher-up:
	docker compose up --build

researcher-up-d:
	docker compose up --build -d

# Pull Docker Hub images (no local build). Repos must be public or `docker login`.
researcher-up-pull:
	docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
	docker compose -f docker-compose.yml -f docker-compose.publish.yml up

researcher-up-pull-d:
	docker compose -f docker-compose.yml -f docker-compose.publish.yml pull
	docker compose -f docker-compose.yml -f docker-compose.publish.yml up -d

researcher-down:
	docker compose down

researcher-logs:
	docker compose logs -f --tail=100 api web

# Academic paper verification (local-only; research-paper/ is gitignored):
#   make -C research-paper paper-quick
#   make -C research-paper mimic-lock
