#!/bin/bash

source path.sh

# Add `--dry_run` below for a config-only sanity check.
python run.py \
    --stages all \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml \
    "$@"
