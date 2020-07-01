#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set="tr_${min_or_max}_${sample_rate}"
dev_set="cv_${min_or_max}_${sample_rate}"
eval_sets="tt_${min_or_max}_${sample_rate} "

./enh.sh \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --fs $sample_rate \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config ./conf/tuning/train_enh_PSM.yaml \
    --speed_perturb_factors "0.9 1.0 1.1" \
    "$@"
