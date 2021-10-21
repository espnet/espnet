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
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# NOTE(kan-bayashi): Make sure that you use text_format=raw
#   if you want to use token_type=char.
token_type=phn

# g2p=g2pk
g2p=g2pk_no_space  # No word sparator

# Default settings for non-vits models
tts_task=tts
feats_extract=fbank
feats_normalize=global_mvn

./tts.sh \
    --tts_task "${tts_task}" \
    --feats_extract "${feats_extract}" \
    --feats_normalize "${feats_normalize}" \
    --local_data_opts "${local_data_opts}" \
    --audio_format wav \
    --lang ko \
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
