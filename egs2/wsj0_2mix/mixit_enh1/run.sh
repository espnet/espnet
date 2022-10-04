#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=min # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set="tr_${min_or_max}_${sample_rate}_w_1spk_utt"
valid_set="cv_${min_or_max}_${sample_rate}"
test_sets="tt_${min_or_max}_${sample_rate} "

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang en \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --enh_config conf/tuning/train_enh_mixit_conv_tasnet.yaml \
    --ref_num 2 \
    --inf_num 4 \
    "$@"
