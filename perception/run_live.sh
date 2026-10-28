#!/usr/bin/env bash
set -eo pipefail
PERCEPTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PERCEPTION_DIR}/activate.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec python -u "${PERCEPTION_DIR}/live_gear36.py" "$@"
