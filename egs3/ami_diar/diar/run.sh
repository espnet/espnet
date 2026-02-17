#!/usr/bin/env bash
# AMI diarization recipe

set -euo pipefail

# Source path configuration
. ./path.sh

# Default configuration
train_config=conf/tuning/train_xeus_conformer_powerset.yaml
infer_config=conf/inference.yaml
metric_config=conf/metric.yaml

# Parse command-line arguments
stages="all"
while [[ $# -gt 0 ]]; do
  case $1 in
    --stages)
      stages="$2"
      shift 2
      ;;
    --train-config)
      train_config="$2"
      shift 2
      ;;
    --infer-config)
      infer_config="$2"
      shift 2
      ;;
    --metric-config)
      metric_config="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--stages STAGES] [--train-config CONFIG] [--infer-config CONFIG] [--metric-config CONFIG]"
      exit 1
      ;;
  esac
done

# Run the pipeline
python run.py \
  --stages "${stages}" \
  --train_config "${train_config}" \
  --infer_config "${infer_config}" \
  --metric_config "${metric_config}"
