#!/usr/bin/env bash
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

score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

opts="--audio_format wav "

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"

# training and inference configuration
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=none
cleaner=none

lang=en # en or kr
if [ "${lang}" == "en" ]; then
    # To suppress recreation, specify wav format
    local_data_opts="--lang english "
    full_lang="english"
else
    local_data_opts="--lang korean "
    full_lang="korean"
fi

train_set=tr_no_dev_${full_lang}
valid_set=dev_${full_lang}
test_sets="dev_${full_lang} eval_${full_lang}"
pitch_extract=dio

./svs.sh \
    --dumpdir dump_${lang} \
    --local_data_opts "${local_data_opts}" \
    --feats_type raw \
    --pitch_extract "${pitch_extract}" \
    --fs "${fs}" \
    --fmax "${fmax}" \
    --fmin "${fmin}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --lang ${lang} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
