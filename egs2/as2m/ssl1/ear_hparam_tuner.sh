#!/bin/bash
set -e
set -u
set -o pipefail

# Default Hyperparameters
TOTAL_STEPS=400000

use_wandb=true

generate_deepspeed_config() {
    local learning_rate=$1
    local warmup_steps=$2

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


wandb_project=EARlarge.PT
# wandb_project=BEATsTokenizerPT 1. change SSL tag here, 2. change ngpu to 3 or 4, 3. change model size in run_ear

for LEARNING_RATE in 5e-4; do
  for WARMUP_STEPS in 40000; do
    for BATCH_BINS in 1600000; do

      SSL_TAG="base.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
      N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

      deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
      deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
      external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_base/beats_tokenizer_iter1_base_tok.tune_lr1.0e-3_warmup40000_bins800000_totalsteps400000/epoch3.pt

      echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
      
      ./run_ear.sh --ngpu 3 --ssl_tag "${SSL_TAG}" \
          --external_tokenizer_model ${external_tokenizer_model} --train_start_iter 1 --train_stop_iter 1 \
          --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
          --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar"
      # sleep 5s

    done
  done
done