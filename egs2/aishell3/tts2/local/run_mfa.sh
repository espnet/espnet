#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_shift=300

./scripts/utils/mfa.sh \
    --language pypinyin_phone  \
    --train true \
    --cleaner tacotron \
    --acoustic_model mandarin_mfa \
    --dictionary mandarin_china_mfa \
    --g2p_model pypinyin_g2p_phone \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --clean_temp true \
    --stage 0 \
    --stop_stage 5 \
    "$@"
