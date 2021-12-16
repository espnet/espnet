#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

asr_config=conf/tuning/train_asr_conformer8.yaml
inference_config=conf/decode_transformer.yaml
lm_config=conf/train_lm.yaml
./asr.sh \
    --ngpu 4 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --fs ${fs} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --local_data_opts "--fs ${fs}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    ${opts} "$@"
