#!/usr/bin/env bash
# Branchformer + speed-perturb + Transformer LM rescoring.
# Pipeline: stages 2-4 (SP data prep) -> 6-8 (LM train) -> 10 (ASR stats) -> 11 (ASR train) -> 12-13 (decode+score with LM).
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_branchformer_sp.yaml
inference_config=conf/decode_asr_branchformer_lm.yaml
lm_config=conf/train_lm_transformer.yaml

use_lm=true
use_wordlm=false
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 2 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm} \
    --use_word_lm ${use_wordlm} \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
