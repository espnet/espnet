#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="mediaspeech_train"
train_dev="mediaspeech_dev"
test_set="mediaspeech_test"

asr_config=conf/train_asr_new.yaml
inference_config=conf/decode_asr.yaml

ngpu=1
nbpe=200

./asr.sh \
    --stage 1 \
    --stop_stage 100 \
    --ngpu ${ngpu} \
    --nj 80 \
    --inference_nj 256 \
    --gpu_inference true \
    --inference_args "--batch_size 1" \
    --use_lm false \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_train_text "data/${train_set}/text" \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --feats_normalize uttmvn \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"