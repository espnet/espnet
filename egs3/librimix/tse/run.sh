#!/bin/bash

source path.sh

python run.py \
    --stages all \
    --training_config conf/tuning/training_td_speakerbeam_16k.yaml \
    --inference_config conf/infererence.yaml \
    --metrics_config conf/metrics.yaml \
    --publication_config conf/publication.yaml \
    --demo_config conf/demo.yaml \
    "$@"
