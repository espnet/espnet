#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="sunda_train" # NOTE no "sunda_" in alam5
train_dev="sunda_dev" # NOTE no "sunda_" in alam5
test_set="sunda_test" # NOTE no "sunda_" in alam5

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

# NOTE started in stage 1 for alam5

./asr.sh \
    --stage 2 \
    --stop_stage 100 \
    --ngpu ${ngpu} \
    --nj 80 \
    --inference_nj 256 \
    --use_lm false \
    --token_type bpe \
    --nbpe 1000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"

