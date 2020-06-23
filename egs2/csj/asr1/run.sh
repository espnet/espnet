#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
dev_set=train_dev
eval_set="eval1 eval2 eval3"

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_set}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --srctexts "data/train_nodev/text" "$@"
