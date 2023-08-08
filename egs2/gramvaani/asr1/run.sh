#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_config="conf/tuning/train_asr_conformer6_n_fft400_hop_length160.yaml"
decode_config="conf/decode_asr.yaml"

./asr.sh \
    --lang "hi" \
    --stage 1 \
    --asr_config $asr_config \
    --inference_config $decode_config \
    --use_lm false \
    --train_set "train100" \
    --valid_set "dev" \
    --test_sets "test" \
    --token_type char  "$@"
