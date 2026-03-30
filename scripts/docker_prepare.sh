#!/usr/bin/env sh
# Run inside compose `prepare` service: train only if model.pkl is missing/empty or FORCE_TRAIN=1.
set -eu
cd /workspace
if [ "${FORCE_TRAIN:-0}" = "1" ] || [ ! -s model.pkl ]; then
  echo "docker_prepare: running training..."
  exec python -m training.train \
    --format longitudinal \
    --data data/raw/ehr_data.csv \
    --model logreg \
    --split-by-patient
fi
echo "docker_prepare: skipping training (model.pkl present). rm model.pkl or FORCE_TRAIN=1 to rebuild."
exit 0
