#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean"

train_config=conf/train_delay_tts.yaml
inference_config=conf/decode_tts_tencent.yaml

codec_opts="--codec_choice inhouse"

./speechlm.sh \
    --task "tts" \
    --fs 16000 \
    --ngpu 1\
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference false \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 0.1 \
    --max_wav_duration 60.0 \
    ${codec_opts} \
    "$@"
