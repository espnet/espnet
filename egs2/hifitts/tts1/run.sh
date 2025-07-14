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

opts=
if [ "${fs}" -eq 44100 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set="tr_no_dev"
valid_set="dev"
test_sets="test"

train_config=/conf/tuning/train_multi_spk_vits.yaml
inference_config=/conf/tuning/decode_vits.yaml

dump_dir=dump/22k
exp_dir=exp/22k


# g2p=g2p_en # Include word separator


./tts.sh \
    --use_sid true \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --dumpdir "${dump_dir}" \
    --expdir "${exp_dir}" \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram  \
    --feats_normalize none  \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${opts} "$@"
