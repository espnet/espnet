#!/usr/bin/env bash
set -euo pipefail
# Runs model upload to huggingface

# wandb_init_args="--use_wandb true --wandb_project DCASE_AAC --wandb_model_log_interval 0"
wandb_init_args=""
other_args="$@"

expdir=/compute/babel-13-33/sbharad2/expdir
dumpdir=/compute/babel-13-33/sbharad2/dumpdir
local_data_opts=/compute/babel-13-33/sbharad2/expdir


expdir=exp/
# asr_tag=pt.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125
# hf_repo=shikhar7ssu/dcase23.aac.pt

asr_tag=ft_lr5e-5.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125
hf_repo=espnet/DCASE23.AudioCaptioning.FineTuned


./asr.sh \
    --asr_tag ${asr_tag} \
    --hf_repo ${hf_repo} \
    --skip_packing false \
    --skip_upload_hf false \
    --lang en \
    --expdir ${expdir} \
    --dumpdir ${dumpdir} \
    --feats_normalize uttmvn \
    --stage 14 \
    --stop_stage 15 \
    --asr_stats_dir ${expdir}/asr_stats_finetune \
    --ngpu 2 \
    --gpu_inference true \
    --nj 8 \
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
    --inference_asr_model valid.acc.ave.pth \
    --asr_args "${wandb_init_args} ${other_args}" \
    --local_data_opts "${local_data_opts}"
