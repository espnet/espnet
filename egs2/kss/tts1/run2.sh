#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Feature related
fs=44100
fmin=80
fmax=22050
n_mels=120
n_fft=2048
n_shift=512
win_length=2048

# Data prep related
text_format=raw  # Use "raw" or "phn". If use "phn", convert to phn in data prep.
local_data_opts=""
local_data_opts+=" --text_format ${text_format}"

dset_suffix=""
if [ "${text_format}" = phn ]; then
    dset_suffix=_phn
fi
train_set=tr_no_dev${dset_suffix}
valid_set=dev${dset_suffix}
test_sets="dev${dset_suffix} eval1${dset_suffix}"

# Config related
##  - you can change the configurations depending on what model to train.
train_config=conf/tuning/train_tacotron2.yaml
inference_config=conf/tuning/decode_tacotron2.yaml

# NOTE(kan-bayashi): Make sure that you use text_format=raw
#   if you want to use token_type=char or token_type=korean_jaso.
token_type=phn
g2p=korean_jaso
cleaner=korean_cleaner

stage=0
stop_stage=10

./tts.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --local_data_opts "${local_data_opts}" \
    --audio_format wav \
    --lang ko \
    --feats_type raw \
    --fs "${fs}" \
    --fmin "${fmin}" \
    --fmax "${fmax}" \
    --n_mels "${n_mels}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type "${token_type}" \
    --g2p "${g2p}" \
    --cleaner "${cleaner}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    "$@"