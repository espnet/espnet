#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

spk=Hokuspokus
train_set=${spk,,}_tr_no_dev
valid_set=${spk,,}_dev
test_sets="${spk,,}_dev ${spk,,}_eval1"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

g2p=espeak_ng_german

./tts.sh \
    --local_data_opts "--spk ${spk}" \
    --audio_format flac.ark \
    --lang de \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    "$@"
