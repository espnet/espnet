#!/bin/bash

source path.sh

python run.py \
    --stages all \
    --train_config conf/tuning/train_e_branchformer.yaml \
    --infer_config conf/infer.yaml \
    --metric_config conf/metric.yaml \
    # --dry_run
