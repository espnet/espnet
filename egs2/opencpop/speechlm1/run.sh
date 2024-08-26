#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=16000
fmin=
fmax=
n_fft=
n_shift=
win_length=


train_set=tr_no_dev
valid_set=dev
test_sets="dev test"

# bpe_opts="--bpemode huggingface --bpemodel allenai/OLMo-1B-hf"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm_svs.sh \
    --stop_stage 5 \
    --task "svs" \
    --data_name opencpop \
    --fs "${fs}" \
    --ngpu 1 \
    --nj 32 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --audio_format "wav" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${codec_opts} \
    "$@"
    