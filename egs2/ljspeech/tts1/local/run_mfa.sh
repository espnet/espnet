#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_shift=256

./scripts/utils/mfa.sh \
    --language english_us_espeak  \
    --train true \
    --cleaner tacotron \
    --g2p_model espeak_ng_english_us_vits \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --clean_temp true \
    "$@"
