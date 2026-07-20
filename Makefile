.PHONY: install test test-web test-all train train-patient train-temporal leak-audit shap docker-smoke cv-report paper-synth paper-exp paper-quick researcher-up researcher-up-d researcher-up-pull researcher-up-pull-d researcher-down researcher-logs mimic-lock

install:
	pip install -r requirements.txt && pip install -e .

test:
	PYTHONPATH=. pytest tests/ -q --tb=short

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

paper-synth:
	PYTHONPATH=. python scripts/generate_paper_synthetic_cohort.py

paper-quick:
	PYTHONPATH=. python scripts/generate_paper_synthetic_cohort.py
	PYTHONPATH=. python scripts/run_paper_experiments.py --data data/raw/paper_synthetic_cohort.csv --out-dir reports/paper --horizon-days 365 --index-strategy column --index-time-col index_time --quick --bootstrap-samples 100
	PYTHONPATH=. python scripts/leakage_audit.py --format longitudinal --data data/raw/paper_synthetic_cohort.csv --split-by-patient --windows 7,30,180 --horizon-days 365 --index-strategy column --index-time-col index_time -o reports/paper/leakage_audit.json

paper-exp:
	PYTHONPATH=. python scripts/run_paper_experiments.py --data data/raw/paper_synthetic_cohort.csv --out-dir reports/paper --horizon-days 365 --index-strategy column --index-time-col index_time --bootstrap-samples 200

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

# Requires local PhysioNet extract at data/processed/mimic_diabetes_cohort.csv (gitignored).
mimic-lock:
	bash scripts/lock_mimic_cohort.sh data/processed/mimic_diabetes_cohort.csv reports/paper/mimic
