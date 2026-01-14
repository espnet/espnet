#!/bin/bash

#set -euo pipefail

python=python3

. ./path.sh
. ../../../utils/parse_options.sh

${python} run.py \
    --stages all \
    --train_config conf/train_asr_transformer_debug.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml \
    "$@"
