#!/usr/bin/env bash
# Train all land-appraisal models with live progress (macOS / Linux).
#   ./scripts/train.sh                     # train with current data (.env: MODEL_VERSION / LAND_PROVINCES)
#   ./scripts/train.sh --prepare           # rebuild cells + OSM features, then train
#   ./scripts/train.sh --only regression,bn
#   LOAD=1 ./scripts/train.sh              # also load results into MongoDB afterwards
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONWARNINGS=ignore
poetry run python -m ml.train_all "$@"

if [[ "${LOAD:-0}" == "1" ]]; then
  poetry run python -m scripts.load_mongo
  if docker compose ps --services --status running 2>/dev/null | grep -qx web; then
    docker compose restart web >/dev/null && echo "web container restarted -> http://localhost:8000"
  fi
fi
