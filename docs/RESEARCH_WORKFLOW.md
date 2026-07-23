# Research workflow (workbench loop)

Research and education only — not a clinical device. This note walks the **claimed research features** end-to-end; it is not a promise of 100% code coverage.

## Step-by-step

| Step | UI | API / artifact |
|------|----|----------------|
| 1. Choose task & dataset | **Datasets** → pick CSV → note task preset | `GET /v1/tasks` · `GET /v1/datasets` |
| 2. Task health check | **Datasets** → Run health (with task) | `GET /v1/datasets/health?path=&task_id=` |
| 3. Train | **Train** → logreg / RF / XGB on demo or upload | `POST /v1/jobs/train` → `reports/runs/<id>/` |
| 4. Trust pack | Written automatically after train | `reports/runs/<id>/trust_pack.json` |
| 5. Leakage audit | **Train** or **Results** → Leakage (bind `run_id`) | `POST /v1/jobs/leakage-audit` → `leakage_audit.json` |
| 6. SHAP | **Results** → Generate SHAP (`run_id`) | `POST /v1/jobs/shap` → `shap_summary.png` |
| 7. External validate | **Results** form or API | `POST /v1/jobs/external-validate` → `external_validation_report.json` |
| 8. Analysis pack | **Analytics** KPI strip + download | `GET /v1/reports/analysis-pack?path=` |
| 9. Methods + ZIP | **Results** download (optional `run_id`) | `GET /v1/reports/methods.md` · `GET /v1/reports/download.zip` |
| 10. Analytics export / print | **Analytics** → Export PNG / Print report | Browser PNG per chart + `window.print()` for appendix figures |

## Minimal API sequence (demo CSV)

```bash
# Workspace + health
curl -s localhost:8000/v1/workspace/status | jq .api_ok
curl -s 'localhost:8000/v1/datasets/health?path=data/demo/ehr_data.csv&task_id=horizon_detection_30d' | jq .health.ready_for_training

# Train (returns job id; poll /v1/jobs/{id})
curl -s -X POST localhost:8000/v1/jobs/train -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","data_format":"longitudinal","model_kind":"logreg","promote":true}'

# After run_id is known:
curl -s -X POST localhost:8000/v1/jobs/leakage-audit -H 'Content-Type: application/json' \
  -d '{"use_artifact":true,"run_id":"RUN_ID"}'
curl -s 'localhost:8000/v1/reports/analysis-pack?path=data/demo/ehr_data.csv' | jq .n_rows
curl -s -X POST localhost:8000/v1/jobs/external-validate -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","run_id":"RUN_ID"}'
curl -s 'localhost:8000/v1/reports/download.zip?run_id=RUN_ID' -o results.zip
```

## Tests

Automated coverage for this loop lives in:

- `tests/test_research_complete_workflow.py` — sequential workflow
- `tests/test_research_compatible.py` — trust pack, leakage, external validate, analysis pack units
- `tests/test_gap_closures.py` — runs detail, fairness, HPO, thresholds

See also [`GAP_CLOSURES.md`](GAP_CLOSURES.md) for shipped vs deferred items.
