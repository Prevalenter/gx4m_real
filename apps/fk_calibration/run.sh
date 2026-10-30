#!/usr/bin/env bash
# Collector: bash apps/fk_calibration/run.sh [--simulate]
# Solver:    bash apps/fk_calibration/run.sh solve <session> [...]
set -euo pipefail
FK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate gear3_control
set -u
if [[ "${1:-}" == solve ]]; then
  shift
  exec python -u "${FK_DIR}/solve.py" "$@"
fi
exec python -u "${FK_DIR}/collect_app.py" "$@"
