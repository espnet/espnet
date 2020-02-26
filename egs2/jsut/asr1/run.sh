#!/bin/bash
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
dev_set=dev
eval_set=eval1

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode_rnn.yaml
lm_config=conf/train_lm.yaml
./asr.sh \
    --token_type char \
    --feats_type raw \
    --fs ${fs} \
    --local_data_opts "--fs ${fs}" \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_set}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
