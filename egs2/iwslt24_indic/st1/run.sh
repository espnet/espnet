#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=en
tgt_lang=hi  # one of hi (Hindi), bn (Bengali), or ta (Tamil)

train_set=train.en-${tgt_lang}
train_dev=dev.en-${tgt_lang}
test_set=tst-COMMON.en-${tgt_lang}

./st.sh \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --local_data_opts "${tgt_lang}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --stage 1 \
    --stop_stage 3

