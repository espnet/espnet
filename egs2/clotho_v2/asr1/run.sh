#!/usr/bin/env bash

set -euo pipefail

timestamp=$(date "+%Y%m%d.%H%M%S")

wandb_init_args="--use_wandb true --wandb_project DCASE_AAC --wandb_model_log_interval 0"
wandb_sweep_args="$@"

timestamp=20240928.224746

./asr.sh \
    --asr_tag ${timestamp} \
    --feats_normalize uttmvn \
    --stage 12 \
    --stop_stage 12 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 10 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set development \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --asr_config conf/dcase.yaml \
    --inference_asr_model 68epoch.pth \
    --asr_args "${wandb_init_args} ${wandb_sweep_args}"
    # --local_score_opts exp/asr_branchformer_raw_en_word/inference_beam_size10_ctc_weight0.3_asr_model_valid.acc.ave \