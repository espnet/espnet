#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
valid_set=dev_clean
test_sets="test_clean"

bpe_opts="--bpemode huggingface --bpemodel EleutherAI/pythia-1b"

# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stop_stage 5 \
    --task "bpe_tts" \
    --data_name librispeech \
    --fs 16000 \
    --ngpu 8 \
    --nj 32 \
    --train_config conf/train_foo.yaml \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --codec_choice ESPnet \
    --codec_hf_model_tag "espnet/amuse_speech_soundstream_16k" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${bpe_opts} \
    "$@"
