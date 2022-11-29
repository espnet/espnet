#!/usr/bin/env bash
# This script is specifically for training with the Montreal Forced Aligner (MFA).
# Importantly, since MFA uses its own custom phonemes, we set token_type to "phn" and g2p to "none".
# Also, default config is for FastSpeech2 since it's the one that uses MFA.
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

train_config=conf/tuning/train_fastspeech2.yaml
inference_config=conf/decode_fastspeech.yaml


./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --cleaner none \
    --token_type phn \
    --g2p none \
    --local_data_opts "--token_type phn --g2p none --use_mfa true" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
