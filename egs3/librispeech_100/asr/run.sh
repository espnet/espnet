#!/bin/bash

source path.sh

python run.py \
    --stages all \
    --train_config conf/tuning/train_e_branchformer.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml \
    # --dry_run
