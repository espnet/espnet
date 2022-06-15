#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euxo pipefail

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


train_config=conf/train.yaml
inference_config=conf/decode.yaml

g2p=g2p_is 

./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner tacotron \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    ${opts} "$@"

## Need to supply the following values in call to run.sh
    # --train_set "${train_set}" \
    # --valid_set "${valid_set}" \
    # --test_sets "${test_sets}" \
    # --expdir "${expdir}" \
    # --srctexts "data/${train_set}/text" \