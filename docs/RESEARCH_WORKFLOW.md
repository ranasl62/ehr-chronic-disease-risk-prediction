# Research workflow (workbench loop)

For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future. This note walks the **claimed research features** end-to-end so you can complete a methods-style study with the tool.

## Recommended: Research wizard

Open the workbench → **Research** (or Home → **Start research wizard**). The wizard walks:

1. Data & task → 2. Health → 3. Train → 4. Trust pack → 5. Leakage → 6. External validation → 7. Export (ZIP / methods.md / Analytics / Predict)

Same artifacts as the manual pages below; prefer the wizard for a first full pass.

**In-app page tours.** Every main nav page has a header **Take tour** control: a lightweight spotlight + tooltip with short documentation (what the page is for, key actions, research/education disclaimer). Use Next / Back / Skip / Finish; optional “Don’t auto-start tours” persists in `localStorage` (`ehr_page_tours_v1`). First visit to a page can auto-start once.

**Health gate.** Horizon / diabetes / readmission tasks need `index_time`. The wizard surfaces blockers loudly and keeps **Next: train** disabled until ready; API train/compare/HPO also run a **task-aware** health check. Prefer `paper_synthetic` for those tasks (or `custom` / `last_event` for the tiny demo).

## Manual step-by-step

| Step | UI | API / artifact |
|------|----|----------------|
| 1. Choose task & dataset | **Datasets** → pick CSV → note task preset | `GET /v1/tasks` · `GET /v1/datasets` |
| 2. Task health check | **Datasets** / wizard → Run health | `GET /v1/datasets/health?path=&task_id=` |
| 3. Train | **Train** → logreg / RF / XGB | `POST /v1/jobs/train` → `reports/runs/<id>/` |
| 4. Trust pack | Automatic after train | Per-run `reports/runs/<id>/trust_pack.json` (file artifact; also on run detail — not `/v1/reports/trust-pack`) |
| 5. Leakage audit | **Train** / **Results** / wizard (`run_id`) | `POST /v1/jobs/leakage-audit` |
| 6. SHAP | **Results** → Generate SHAP (`run_id`) | `POST /v1/jobs/shap` → `shap_summary.png` |
| 7. External validate | **Results** / wizard | `POST /v1/jobs/external-validate` |
| 8. Analysis pack | **Analytics** KPIs + download | `GET /v1/reports/analysis-pack?path=` |
| 9. Methods + ZIP | **Results** / wizard download | `GET /v1/reports/methods.md` · `download.zip` |
| 10. Paper figures | **Analytics** → ROC/PR/calibration + PNG / Print | Curves in `evaluation_report.json`; `GET /v1/reports/curves` |
| 11. Predict session | **Predict** → Export session JSON | Features + score for lab notebooks (not care) |

**Curves & CIs.** After a **retrain**, hold-out ROC / PR / calibration points are stored under `evaluation_report.json` → `curves`, plotted on Analytics (Chart.js), and served by `GET /v1/reports/curves`. When the hold-out supports it, percentile bootstrap CIs for ROC/PR-AUC appear as `bootstrap_cis` / a quality note. Tiny or single-class hold-outs correctly report AUC as n/a.

## Minimal API sequence (demo CSV)

```bash
# Workspace + health
# Tiny demo lacks index_time — use custom / last_event, or paper_synthetic for horizon tasks.
curl -s localhost:8000/v1/workspace/status | jq .api_ok
curl -s 'localhost:8000/v1/datasets/health?path=data/demo/ehr_data.csv&task_id=custom' | jq .health.ready_for_training
curl -s 'localhost:8000/v1/datasets/health?path=data/raw/paper_synthetic_cohort.csv&task_id=horizon_detection_30d' | jq .health.ready_for_training

# Train (returns job id; poll /v1/jobs/{id})
curl -s -X POST localhost:8000/v1/jobs/train -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","data_format":"longitudinal","model_kind":"logreg","promote":true}'

# After run_id is known:
curl -s -X POST localhost:8000/v1/jobs/leakage-audit -H 'Content-Type: application/json' \
  -d '{"use_artifact":true,"run_id":"RUN_ID"}'
curl -s 'localhost:8000/v1/reports/analysis-pack?path=data/demo/ehr_data.csv' | jq .n_rows
curl -s -X POST localhost:8000/v1/jobs/external-validate -H 'Content-Type: application/json' \
  -d '{"data_path":"data/demo/ehr_data.csv","run_id":"RUN_ID"}'
curl -s 'localhost:8000/v1/reports/curves' | jq '.curves.roc.fpr | length'
curl -s 'localhost:8000/v1/reports/thresholds' | jq .
curl -s 'localhost:8000/v1/reports/download.zip?run_id=RUN_ID' -o results.zip
```

## Tests

- `tests/test_research_complete_workflow.py` — sequential workflow (TestClient)
- `tests/test_e2e_features.py` — live UI→API e2e (`@pytest.mark.e2e`)
- `tests/test_eval_curves.py` / `tests/test_reports_curves.py` — curve payloads + API
- `web/e2e` — optional Playwright smoke (`cd web && npm run e2e`)

See also [`GAP_CLOSURES.md`](GAP_CLOSURES.md) · [`external_validation.md`](external_validation.md).
