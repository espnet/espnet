#!/bin/bash

source path.sh

python run.py \
    --stages all \
    --train_config conf/tuning/train_td_speakerbeam_16k.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml \
    --publish_config conf/publish.yaml \
    --demo_config conf/demo.yaml
