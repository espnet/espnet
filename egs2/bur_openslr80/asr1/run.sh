#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="bur_train"
train_dev="bur_dev"
test_set="bur_test"

asr_config=conf/train_asr_hubert_transformer_adam_specaug.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml
ngpu=1

./asr.sh \
    --stage 1 \
    --stop_stage 100 \
    --ngpu ${ngpu} \
    --nj 80 \
    --inference_nj 256 \
    --inference_asr_model valid.acc.best.pth \
    --gpu_inference false \
    --inference_args "--batch_size 1" \
    --use_lm true \
    --token_type bpe \
    --nbpe 150 \
    --feats_type raw \
    --feats_normalize uttmvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}"\
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --lm_dev_text "data/${train_dev}/text" \
    --lm_test_text "data/${test_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"
