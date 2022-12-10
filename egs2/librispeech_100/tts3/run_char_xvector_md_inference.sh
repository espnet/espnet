#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000 # original 24000
n_fft=2048
n_shift=300
win_length=1200

tag="tts_16k_char_xvector_pretrain_no_finetune"

train_set="train_clean_100"
valid_set="dev_clean"
# test_sets="dev_clean"
test_sets="train_clean_360"

train_config=conf/tuning/train_transformer_xvector_md_pretrain_no_finetune.yaml
inference_config=conf/tuning/decode_ctc_bs1.yaml
inference_asr_config=conf/decode_asr.yaml


./tts.sh \
    --ngpu 1 \
    --stage 7 \
    --use_multidecoder true \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector true \
    --token_type char \
    --cleaner none \
    --tag "${tag}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_config "${inference_asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_model valid.loss.best.pth \
    --srctexts "data/${train_set}/text" \
    --audio_format "wav" "$@"
