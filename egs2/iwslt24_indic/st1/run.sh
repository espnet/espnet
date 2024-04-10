#!/usr/bin/env bash

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
    --stage 1 \
    --stop_stage 1

