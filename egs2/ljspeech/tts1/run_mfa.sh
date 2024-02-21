#!/usr/bin/env bash
# This script is specifically for training with phonemes and durations prepared by Montreal Forced Aligner (MFA).

set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256

# We should merge local/run_mfa.sh here, but parsing two different sets of options is confusing.

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

# Use phonemized data sets
train_set=tr_no_dev_phn
valid_set=dev_phn
test_sets="dev_phn eval1_phn"

# Default config is for FastSpeech2 since it's the one that uses MFA.
train_config=conf/tuning/train_fastspeech2.yaml
inference_config=conf/tuning/decode_fastspeech.yaml

# write_collected_feats makes training 8-10x faster
./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --cleaner none \
    --token_type phn \
    --g2p none \
    --write_collected_feats true \
    --teacher_dumpdir "dump/raw" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --stage 2 \
    ${opts} "$@"
