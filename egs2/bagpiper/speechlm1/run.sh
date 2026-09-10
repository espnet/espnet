#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./train.sh \
    --train-config conf/train.yaml \
    --output-dir exp/warmup \
    --wandb-project bagpiper \
    "$@"
