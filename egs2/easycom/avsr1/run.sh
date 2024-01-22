#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="val"
test_set="test"

av_hubert_model="large" #select large or base
include_wearer=False    #False is normal for easycom AVSR
with_LRS3=True
noise_augmentation=True
config=conf/train_avsr_avhubert_${av_hubert_model}_with_lrs3_noise.yaml

if [ ${with_LRS3} ]; then
    train_set=${train_set}_with_LRS3
    valid_set=${valid_set}_with_LRS3
    test_set=${test_set}_with_LRS3
fi

./asr.sh \
    --lang en \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_set} \
    --feats_type extracted \
    --local_data_opts "${av_hubert_model} ${include_wearer} ${with_LRS3} ${noise_augmentation}"\
    --token_type bpe \
    --nbpe 1000 \
    --bpe_train_text "data/${train_set}/text" \
    --use_lm false \
    --asr_config ${config} \
    --ngpu 1  "$@"
