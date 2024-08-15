#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_clean_100
valid_set=dev_clean
test_sets="test_clean"

train_config="conf/train_delay.yaml"

ssl_opts="--ssl_checkpoint_path exp/kmeans_xues/38epoch.pth --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl --ssl_nlayer 16"

./speechlm.sh \
    --task "ssl_asr" \
    --data_name librispeech_100 \
    --fs 16000 \
    --ngpu 1 \
    --nj 8 \
    --train_config ${train_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${ssl_opts} \
    "$@"
