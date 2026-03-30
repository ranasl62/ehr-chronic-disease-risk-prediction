# Train and evaluate

Run training for this project and confirm artifacts.

1. From repo root with venv active and `pip install -e .` (or `PYTHONPATH=.`).
2. Longitudinal demo (default 7/30/180d windows):

```bash
python -m training.train --format longitudinal --data data/raw/ehr_data.csv --model logreg
```

3. Confirm `model.pkl`, `reports/evaluation_report.json`, `reports/feature_importance.json`, and `reports/calibration_holdout.png` exist.
4. If changing feature schema, update `dashboard/app.py` / API clients to use `GET /v1/model/schema`.

Report ROC-AUC/PR-AUC/Brier from console or JSON; note small-demo overfitting risk.
