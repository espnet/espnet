#!/bin/bash

set -euo pipefail

python=python3

. ./path.sh
. ../../../utils/parse_options.sh

${python} run.py \
    --stages all \
    --train_config conf/train.yaml \
    --infer_config conf/inference.yaml \
    --measure_config conf/measure.yaml \
    "$@"
