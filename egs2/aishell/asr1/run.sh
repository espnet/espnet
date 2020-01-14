#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_sets="test "

# speed perturbation related
speed_perturb=True  # train_set will be "${train_set}_sp" if speed_perturb = True
perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nbpe 5000 \
    --token_type char \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text" "$@" \
    --speed_perturb "${speed_perturb}" \
    --perturb_factors "${perturb_factors}"
