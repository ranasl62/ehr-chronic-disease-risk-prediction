# Data guide

**Contact:** [support@larucare.com](mailto:support@larucare.com)

## Synthetic vs real (same pipeline)

| Mode | Examples | Credentials |
|------|----------|-------------|
| **Synthetic / demo** | `data/demo/ehr_data.csv`, `data/demo/sample_ehr.csv`, `data/raw/paper_synthetic_cohort.csv` | None |
| **Real BYO** | Your CSV/XLSX/JSON via map wizard | Your IRB/DUA |
| **MIMIC** | Credentialed local extract | PhysioNet |
| **OMOP / FHIR** | Subset adapters → same CSV schema | Your warehouse |

Tag datasets with `source_type`: `synthetic` | `demo` | `byo` | `mimic` | `omop` | `fhir`.

## Longitudinal contract (minimum)

| Column | Role |
|--------|------|
| `patient_id` | Subject key (aliases: `subject_id`, `person_id`, …) |
| `timestamp` | Event time |
| `label` | Outcome (or map from `outcome` / …) |
| `index_time` | Recommended for horizon-safe tasks |

Numeric labs/vitals as available. See [`docs/data_sources_and_schema.md`](docs/data_sources_and_schema.md).

## Supported ingest

| Mode | How |
|------|-----|
| Bundled demo CSV | Datasets browse |
| File upload | CSV, TSV, JSON, XLSX |
| Column map | `POST /v1/datasets/map-preview` + `map-import` |
| Form / JSON rows | Datasets form tab |
| SQL (read-only `SELECT`) | Datasets SQL tab |
| OMOP / FHIR subset | API adapters |

Uploads land under `data/uploads/` (gitignored). Bundled teaching fixtures are under `data/demo/`; older `data/raw/ehr_data.csv` references resolve to the demo file. **Never commit PHI.**

## Failure checklist

- Missing `patient_id` → map `member_id` / `person_id` / `subject_id`
- Missing label → map `outcome` / `chronic_disease` / `label`
- Bad dates → fix `timestamp` / `service_date`
- Horizon tasks without `index_time` → leakage risk MEDIUM (health warning)
- Tiny cohort (&lt;50 patients) → unstable metrics warning

## Dataset health

```bash
curl 'http://127.0.0.1:8000/v1/datasets/health?path=data/demo/ehr_data.csv'
```

## Task YAML

See `tasks/*.yaml` and Config Center (`/config`). Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Limits

[`LIMITATIONS.md`](LIMITATIONS.md) · Feedback: [`docs/HOW_IT_HELPS.md`](docs/HOW_IT_HELPS.md).
