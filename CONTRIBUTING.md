# Contributing

Thanks for helping improve this **research** EHR prediction framework.

**Maintainer:** Md Rana Hossain · [support@larucare.com](mailto:support@larucare.com)

## Feedback first

Not ready to open a PR? Email either address or use GitHub Issues.  
What helps most: steps to reproduce, environment (Docker vs pip), and **no PHI**.  
Guide: [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md).

## Ground rules

1. **Temporal integrity** — features must not use post-index information; patient splits when claiming patient-level generalization.
2. **No PHI in git** — only demo/synthetic data under `data/raw/`.
3. **Honest claims** — do not market LSTM stub as production, or full FHIR/OMOP servers / Model Hub, until implemented (`LIMITATIONS.md`).
4. **Tests** — `PYTHONPATH=. pytest tests/ -q` after API / train / schema changes.
5. **Human authorship** — see `AUTHORS.md`.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
ehr-ai init
```

UI: `cd web && npm install && npm start` (proxies to API on :8000).

## Good first issues

- Add a task preset YAML under `tasks/` (copy `tasks/custom.yaml`)
- Docs typos / clarify install steps
- Refresh Angular screenshots per `scripts/capture_ui_screenshots.md`
- Extra unit tests for a single `openhealth/` module edge case
- Improve error messages on dataset health blockers

## Pull requests

- Prefer small PRs focused on one layer (task, health, compare, docs).
- **New API route = new test** (TestClient happy path + 4xx).
- Keep README / ARCHITECTURE / LIMITATIONS in sync when adding public surfaces.
- Run `PYTHONPATH=. pytest tests/ -q` before opening a PR.
