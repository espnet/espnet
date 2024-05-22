#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev_clean
test_sets="dev_clean test_clean"

train_config=conf/train_multiscale.yaml
inference_config=conf/decode_encodec.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en

# Note(Jinchuan): We only select audio range from 3s to 30s since:
#                 (1) The speech prompt is 3s
#                 (2) We limit the longest audio to 30s to avoid
#                     some corner cases in memeory
./speechlm.sh \
    --task "tts" \
    --fs 16000 \
    --ngpu 8 \
    --nj 64 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --codec_choice inhouse \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    "$@"
