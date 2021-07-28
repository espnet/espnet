#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Feature related
fs=24000
n_fft=2048
n_shift=300
win_length=1200

# Data prep related
text_format=phn  # Use "raw" or "phn". If use "phn", convert to phn in data prep.
local_data_opts=""
local_data_opts+=" --text_format ${text_format}"
if [ "${text_format}" = phn ]; then
    local_data_opts+=" --g2p espeak_ng_russian"
fi

dset_suffix=""
if [ "${text_format}" = phn ]; then
    dset_suffix=_phn
fi
train_set=tr_no_dev${dset_suffix}
valid_set=dev${dset_suffix}
test_sets="dev${dset_suffix} eval1${dset_suffix}"

# config related
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# NOTE(kan-bayashi): Make sure that you use text_format=raw
#   if you want to use token_type=char.
token_type=phn

# NOTE(kan-bayashi): On-the-fly with Espeak is really slow,
#   so we convert text into phn in data prep stage via
#   --text_format=phn and use g2p=none for training.
# g2p=espeak_ng_russian
g2p=none

./tts.sh \
    --local_data_opts "${local_data_opts}" \
    --audio_format wav \
    --lang ru \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type "${token_type}" \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    "$@"
