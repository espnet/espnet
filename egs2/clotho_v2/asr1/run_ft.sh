#!/usr/bin/env bash
set -euo pipefail
# Runs fine-tuning

# wandb_init_args="--use_wandb true --wandb_project DCASE_AAC --wandb_model_log_interval 0"
wandb_init_args=""
other_args="$@"

pre_trained_model_path=exp/asr_pt.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125/valid.acc.ave_5best.pth

./asr.sh \
    --asr_tag ft_lr5e-5.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125 \
    --feats_normalize uttmvn \
    --stage 11 \
    --stop_stage 13 \
    --asr_stats_dir exp/asr_stats_finetune \
    --ngpu 2 \
    --gpu_inference true \
    --nj 20 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set development_clotho \
    --valid_set validation \
    --test_sets "evaluation" \
    --asr_config conf/beats_bart_ft.yaml \
    --pretrained_model ${pre_trained_model_path} \
    --inference_asr_model valid.acc.ave_5best.pth \
    --asr_args "${wandb_init_args} ${other_args}" \
    --local_score_opts exp/asr_ft.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125/inference_beam_size10_ctc_weight0.0_hugging_face_decoderTrue_asr_model_valid.acc.ave_5best
