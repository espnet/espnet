#!/usr/bin/env bash
# Runs pre-training
set -euo pipefail

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=pt.cleanacaps.bs512-warmup8k.load_bart.3ladapter.${timestamp}

wandb_init_args="--use_wandb true --wandb_entity shikhar --wandb_name ${mynametag} --wandb_project DCASE_AAC --wandb_model_log_interval 0"
other_args="$@"

expdir=/compute/babel-13-33/sbharad2/expdir
dumpdir=/compute/babel-13-33/sbharad2/dumpdir_clean
local_data_opts=/compute/babel-13-33/sbharad2/expdir
asr_speech_fold_length=4800 # 480000/16000 = 30 seconds

./asr.sh \
    --asr_tag ${mynametag} \
    --expdir ${expdir} \
    --dumpdir ${dumpdir} \
    --asr_speech_fold_length ${asr_speech_fold_length} \
    --local_data_opts ${local_data_opts} \
    --feats_normalize uttmvn \
    --stage 11 \
    --stop_stage 13 \
    --ngpu 2 \
    --gpu_inference true \
    --nj 10 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set pretrain \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --asr_config conf/beats_bart_pt.yaml \
    --inference_asr_model valid.acc.best.pth \
    --asr_args "${wandb_init_args} ${other_args}" \
    --local_score_opts "exp/asr_${mynametag}/inference_beam_size10_ctc_weight0.0_hugging_face_decoderTrue_asr_model_valid.acc.best"
