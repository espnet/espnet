#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

tgt_lang=de

train_set=train.en-${tgt_lang}
train_dev=dev.en-${tgt_lang}
test_set="tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

task=mt # mt will use tc as src_text

./st.sh \
    --local_data_opts "${tgt_lang} ${task}" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --bpemode huggingface \
    --bpemodel bert-base-cased \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" "$@"
