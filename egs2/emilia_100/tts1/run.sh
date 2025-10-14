#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256
win_length=null

opts="--audio_format flac "

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"
tts_task=gan_tts

train_config=conf/train.yaml
inference_config=conf/decode.yaml

cleaner=tacotron
g2p=g2p_en

./tts.sh \
    --ngpu 4 \
    --lang en \
    --feats_type raw \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_spk_embed true \
    --spk_embed_tool python \
    --spk_embed_tag xvector \
    --tts_task "${tts_task}" \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
