#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./train.sh \
    --train-config conf/train_pretrain.yaml \
    --output-dir exp/pretrain \
    --wandb-project bagpiper \
    "$@"
