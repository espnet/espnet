#!/usr/bin/env bash
# Decode + score the EXISTING e_branchformer_scratch ASR model using the
# WARM-STARTED LM (exp/lm_train_lm_transformer_zh_char_warmstart) instead of
# the from-scratch LM, to isolate "warm-start LM" as the only variable.
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr_branchformer_lm.yaml
speed_perturb_factors="0.9 1.0 1.1"
asr_tag=e_branchformer_scratch
warmstart_lm_exp=exp/lm_train_lm_transformer_zh_char_warmstart

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 2 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm true \
    --use_word_lm false \
    --lm_exp "${warmstart_lm_exp}" \
    --asr_config "${asr_config}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
