.PHONY: install test train train-patient train-temporal leak-audit shap docker-smoke cv-report

install:
	pip install -r requirements.txt && pip install -e .

test:
	PYTHONPATH=. pytest tests/ -q --tb=short

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
