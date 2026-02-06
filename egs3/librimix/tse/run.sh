#!/bin/bash

source path.sh

python run.py \
    --stages all \
    --train_config conf/tuning/train_td_speakerbeam_16k.yaml \
    --infer_config conf/inference.yaml \
    --metric_config conf/metric.yaml
