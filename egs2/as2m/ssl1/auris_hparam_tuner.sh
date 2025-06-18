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


wandb_project=Auris.PT
# wandb_project=BEATsTokenizerPT 1. change SSL tag here, 2. change ngpu to 3 or 4, 3. change model size in run_ear


# Auris-LARGE - 1kvocab
for LEARNING_RATE in 1.0e-4; do
  for WARMUP_STEPS in 40000; do
    for BATCH_BINS in 800000; do
      for mixup in 5; do
        for contrastive_weight in 5.0; do
          ITER=1
          SSL_TAG="auris.large${ITER}.contra${contrastive_weight}.mixup${mixup}.lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
          # SSL_TAG="auris.large${ITER}.mixup${mixup}.lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
          N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

          deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
          deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
          external_tokenizer_model=/work/nvme/bbjs/sbharadwaj/model_checkpoints/ear_large/beats_tokenizer_iter2_large_tok2.tune_lr5.0e-4_warmup20000_bins300000_totalsteps100000/epoch4.pt

          echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
          
          cp conf/auris_large.yaml conf/auris_large.mixup${mixup}.contra${contrastive_weight}.yaml
          sed -i "s/mixup_probability: 0.0/mixup_probability: 0.${mixup}/g" conf/auris_large.mixup${mixup}.contra${contrastive_weight}.yaml
          sed -i "s/contrastive_loss_weight: 0.5/contrastive_loss_weight: ${contrastive_weight}/g" conf/auris_large.mixup${mixup}.contra${contrastive_weight}.yaml

          ./run_auris.sh --ngpu 4 --ssl_tag "${SSL_TAG}" --num_nodes 1 \
              --external_tokenizer_model ${external_tokenizer_model} \
              --train_start_iter ${ITER} --train_stop_iter ${ITER} \
              --train_config conf/auris_large.mixup${mixup}.contra${contrastive_weight}.yaml \
              --tokenizer_train_config conf/tok_ear_large.yaml \
              --tokenizer_inference_batch_size 500 \
              --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
              --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
        # sleep 5s
        done
      done
    done
  done
done

wait


# Auris LARGE-2kvocab
# TOTAL_STEPS=800000
# for LEARNING_RATE in 1.0e-4; do
#   for WARMUP_STEPS in 40000; do
#     for BATCH_BINS in 1600000; do
#       ITER=0
#       SSL_TAG="large${ITER}.2kvocab.tune_lr${LEARNING_RATE}_warmup${WARMUP_STEPS}_bins${BATCH_BINS}_totalsteps${TOTAL_STEPS}"
#       N_EPOCH=$(awk "BEGIN {print int($TOTAL_STEPS * ($BATCH_BINS / (998 + 500)) / 7220000)}")

#       deepspeed_config_json_str=$(generate_deepspeed_config "$LEARNING_RATE" "$WARMUP_STEPS")
#       deepspeed_config_json_str=$(echo "$deepspeed_config_json_str" | base64 -w 0)
#       echo "Starting run with N_EPOCH=${N_EPOCH}, SSL_TAG=${SSL_TAG}, LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, BatchBins=${BATCH_BINS}"
      
#       ./run_beatsv2_large_2kvocab.sh --ngpu 4 --ssl_tag "${SSL_TAG}" \
#           --train_start_iter ${ITER} --train_stop_iter ${ITER} \
#           --stage 7 --stop_stage 7 \
#           --n_targets 2048 \
#           --train_config conf/ear_large2k.yaml \
#           --tokenizer_train_config conf/tok_ear_large2k.yaml \
#           --tokenizer_inference_config conf/tokenizer_inf_ear_large2k.yaml \
#           --tokenizer_inference_batch_size 40 \
#           --beats_args "--batch_bins ${BATCH_BINS} --max_epoch ${N_EPOCH} --deepspeed_config '${deepspeed_config_json_str}' \
#           --use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${SSL_TAG} --wandb_entity shikhar" &
#       # sleep 5s
#     done
#   done
# done

# wait


# TODO(mixup + auris)
wait