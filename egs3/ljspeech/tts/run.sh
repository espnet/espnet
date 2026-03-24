#!/bin/bash

python=python3
stages="create_dataset collect_stats train infer"
training_config=conf/training.yaml
inference_config=conf/inference.yaml

. ./path.sh
. ../../../egs2/TEMPLATE/asr1/utils/parse_options.sh

${python} run.py \
    --stages ${stages} \
    --training_config "${training_config}" \
    --inference_config "${inference_config}" \
    "$@"
