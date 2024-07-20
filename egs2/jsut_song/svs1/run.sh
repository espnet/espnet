#!/usr/bin/env bash

# Reference from ESPnet's egs2/nit_song070/svs1/run.sh
# https://github.com/espnet/espnet/blob/master/egs2/nit_song070/svs1/run.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=24000
fmin=80
fmax=7600
n_fft=2048
n_shift=300
win_length=1200

score_feats_extract=frame_score_feats   # frame_score_feats | syllable_score_feats

opts="--audio_format wav "

train_set=train
valid_set=dev
test_sets="dev eval"

# training and inference configuration
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=none
cleaner=none

./svs.sh \
    --lang jp \
    --local_data_opts "--stage 0" \
    --feats_type raw \
    --fs "${fs}" \
    --fmin "${fmin}" \
    --fmax "${fmax}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
