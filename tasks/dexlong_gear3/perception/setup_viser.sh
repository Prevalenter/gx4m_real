#!/usr/bin/env bash
set -eo pipefail
PERCEPTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PERCEPTION_DIR}/activate.sh"
python -m pip install -r "${PERCEPTION_DIR}/viser-requirements.txt"
python -m pip install --no-deps -e "${PERCEPTION_DIR}/../../.."
