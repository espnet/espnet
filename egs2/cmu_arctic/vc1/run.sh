#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000      # sampling frequency
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length=null

train_config=conf/train.yaml
inference_config=conf/decode.yaml

srcspk=clb
trgspk=slt

src_train_set=${srcspk}_train
src_valid_set=${srcspk}_dev
src_test_sets="${srcspk}_eval"
trg_train_set=${trgspk}_train
trg_valid_set=${trgspk}_dev
trg_test_sets="${trgspk}_eval"

local_data_opts="${srcspk} ${trgspk}"

# g2p=g2p_en # Include word separator
g2p=g2p_en_no_space # Include no word separator

./vc.sh \
    --ngpu 1 \
    --lang en \
    --feats_type raw \
    --audio_format wav \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector false \
    --token_type phn \
    --cleaner tacotron \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --src_train_set "${src_train_set}" \
    --src_valid_set "${src_valid_set}" \
    --src_test_sets "${src_test_sets}" \
    --trg_train_set "${trg_train_set}" \
    --trg_valid_set "${trg_valid_set}" \
    --trg_test_sets "${trg_test_sets}" \
    --srctexts "data/${src_train_set}/text" \
    "$@"
