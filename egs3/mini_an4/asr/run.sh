#!/bin/bash

#set -euo pipefail

python=python3

. ./path.sh
. ../../../utils/parse_options.sh

${python} run.py \
    --stages all \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    "$@"
