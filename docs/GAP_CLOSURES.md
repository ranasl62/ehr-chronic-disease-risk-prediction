# Gap closures (research workbench)

Shipped in this pass vs intentionally deferred. Framing: research / education only — not a clinical device.

## Shipped

| Area | What |
|------|------|
| Per-run trust pack | `trust_pack.json` under `reports/runs/<id>/`; leakage/SHAP/ext-val bind to `run_id`; promote syncs trust artifacts; ZIP/`methods.md` accept `run_id` |
| External validation job | `POST /v1/jobs/external-validate` + Results UI form |
| Analysis pack | `GET /v1/reports/analysis-pack` + Analytics KPIs/download |
| Leakage ICD warnings | `warnings[]` when ICD/diagnosis-style feature names appear |
| Horizon / teaching tasks | `horizon_detection_30d`, `teaching_leaky_contrast` + demo CSV |
| Task-aware health | `GET /v1/datasets/health?task_id=` enforces required columns |
| Experiment browser | `GET /v1/runs`, `GET /v1/runs/{id}`, promote; Results UI list / open / compare / promote |
| Fairness panel | `GET /v1/reports/fairness`, Results “Run fairness” + group table (age bands / TPR·FPR); summary + ZIP |
| Light HPO | `POST /v1/jobs/hpo` + Train optional light grid; Results best-trial card + trials table (`n/a` for unavailable metrics); `reports/hpo_report.json` |
| Demo dataset catalog | Teaching fixtures under `data/demo/`; Train defaults to `data/demo/ehr_data.csv`; legacy raw paths resolve as fallbacks |
| Dataset browse / delete | Demo toggle, tabular selection, multi-delete; missing allowed paths return idempotent `already_absent` |
| Job UX | Cancel (queued/best-effort), recent jobs list on Train, clearer status |
| Task clarity | Task description + `required_columns` (readmission_30d highlights `index_time`) |
| Auth empty states | Global 401 banner → Config; Predict already CTAs to Train when no schema |
| LSTM | Not selectable in Train/Config; meta lists under unsupported |
| Thresholds | `GET /v1/reports/thresholds` + Results table (operating points) |
| Docs | This note · [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) · brief website updates (ui-analytics / features) |
| Analytics export / print | Per-chart PNG + print/PDF layout on Analytics (research appendix figures; not clinical reporting) |

## Deferred

| Item | Why |
|------|-----|
| Full Optuna / AutoML | Out of scope; light grid only |
| Hard-kill of running sklearn fits | Best-effort cancel flag only |
| LSTM / Transformers path | Explicit non-goal (`LIMITATIONS.md`) |
| Clinical decision thresholds | Research operating points only |
| Multi-tenant auth / SaaS | Local optional `API_KEY` only |
