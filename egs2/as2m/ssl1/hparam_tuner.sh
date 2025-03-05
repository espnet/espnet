#!/bin/bash
set -e
set -u
set -o pipefail

# Default Hyperparameters
N_EPOCH=50
TOTAL_STEPS=50000

# Wandb arguments
use_wandb=true
wandb_project=BEATsPTi0
wandb_entity=shikhar

# Function to generate DeepSpeed config
generate_deepspeed_config() {
    local learning_rate=$1
    local warmup_steps=$2
    local batch_bins=$3

    cat <<EOF
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": ${learning_rate},
      "betas": [0.9, 0.98],
      "eps": 1e-12,
      "weight_decay": 1.0e-2,
      "adam_w_mode": true
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_type": "linear",
      "total_num_steps": ${TOTAL_STEPS},
      "warmup_num_steps": ${warmup_steps},
      "warmup_max_lr": ${learning_rate},
      "warmup_min_lr": 1.0e-6
    }
  },
  "wall_clock_breakdown": false,
  "steps_per_print": 3000
}
EOF
}


for LEARNING_RATE in 5.0e-4 1.0e-3 ; do
  for WARMUP_STEPS in 2000 4000 8000; do
    for BATCH_BINS in 2886000; do

      SSL_TAG="tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}"
      # SSL_TAG=testing_args_norun

      deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS" "$BATCH_BINS")
      deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)

      echo "Starting run with SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
      
      ./run.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
          --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
          --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity ${wandb_entity}" &

    done
  done
done