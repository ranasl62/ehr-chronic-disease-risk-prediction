#!/usr/bin/env bash
# Lock a credentialed MIMIC (or local) longitudinal extract for paper Results.
# Does NOT download MIMIC. Keep restricted CSVs out of git (data/processed/ is gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IN="${1:-data/processed/mimic_diabetes_cohort.csv}"
OUT_DIR="${2:-reports/paper/mimic}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$IN" ]]; then
  echo "Missing extract: $IN"
  echo "1) Complete PhysioNet credentialing"
  echo "2) Run SQL from sql/feature_queries.sql against your MIMIC Postgres"
  echo "3) Normalize to longitudinal contract (scripts/normalize_longitudinal_csv.py)"
  echo "4) Ensure columns: patient_id,timestamp,index_time,...,label"
  echo "See docs/mimic_lock_checklist.md and docs/mimic_extract_splits_runbook.md"
  exit 1
fi

export PYTHONPATH=.
python scripts/validate_training_data.py --format longitudinal "$IN"
python scripts/leakage_audit.py \
  --format longitudinal --data "$IN" \
  --split-by-patient --windows 7,30,180 \
  --horizon-days 365 --index-strategy column --index-time-col index_time \
  -o "$OUT_DIR/leakage_audit.json"

python scripts/run_paper_experiments.py \
  --data "$IN" \
  --out-dir "$OUT_DIR" \
  --horizon-days 365 \
  --index-strategy column \
  --index-time-col index_time \
  --split-by-patient \
  --bootstrap-samples 1000

# Primary artifact for SHAP / fairness (XGBoost calibrated multi-window)
PRIMARY="$OUT_DIR/model_xgboost_cal_w7-30-180.pkl"
if [[ -f "$PRIMARY" ]]; then
  python scripts/explain_shap.py --artifact "$PRIMARY" --out "$OUT_DIR/shap_summary.png"
  if [[ -f data/processed/mimic_diabetes_groups.csv ]]; then
    python scripts/fairness_report.py \
      --artifact "$PRIMARY" \
      --subgroups data/processed/mimic_diabetes_groups.csv \
      --group-column sex \
      -o "$OUT_DIR/fairness_sex.json" || true
  fi
fi

# Record lock metadata (no PHI)
python - <<PY
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone
p = Path("$IN")
h = hashlib.sha256(p.read_bytes()).hexdigest()
meta = {
    "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_path": str(p.resolve()),
    "data_sha256": h,
    "protocol": "docs/mimic_lock_checklist.md",
    "out_dir": "$OUT_DIR",
}
Path("$OUT_DIR/cohort_lock.json").write_text(json.dumps(meta, indent=2))
print("Wrote", "$OUT_DIR/cohort_lock.json", "sha256=", h[:16], "...")
PY

echo "MIMIC lock complete → $OUT_DIR"
