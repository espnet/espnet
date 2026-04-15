#!/bin/bash

python=python3

. ./path.sh
. ../../../utils/parse_options.sh

${python} run.py \
    --stages create_dataset collect_stats train infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    "$@"
