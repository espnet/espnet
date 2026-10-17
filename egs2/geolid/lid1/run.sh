#!/usr/bin/env bash
set -euo pipefail
exec bash ./run_voxlingua107_only.sh --lid_config conf/train.yaml "$@"
