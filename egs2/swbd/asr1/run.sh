#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval2000"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="1.1 0.9 1.0"

bpe_train_text=dump/fbank_pitch/train_nodup_sp/text
lm_train_text=data/lm_train.txt

# NOTE: The default settings require 8 GPUs with 32 GB memory
./asr.sh \
    --ngpu 8 \
    --token_type bpe \
    --nbpe 2000 \
    --bpe_train_text ${bpe_train_text} \
    --lm_train_text ${lm_train_text} \
    --feats_type fbank_pitch \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_lm valid.loss.best.pth \
    --lm_config "${lm_config}" \
    --score_opts "-s" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    "$@"
