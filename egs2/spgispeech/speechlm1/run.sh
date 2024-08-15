#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

norm="_unnorm"
train_set=train_nodev${norm}
valid_set=dev_4k${norm}
test_sets="dev_4k${norm} val${norm}"

bpe_opts="--bpemode huggingface --bpemodel allenai/OLMo-1B-hf"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"

# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stop_stage 5 \
    --task "plain_bpe_tts" \
    --data_name spgispeech \
    --fs 16000 \
    --ngpu 8 \
    --nj 32 \
    --train_config conf/train_foo.yaml \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${bpe_opts} ${codec_opts} \
    "$@"
