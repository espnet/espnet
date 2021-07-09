#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="train"
train_dev="dev"
test_set="$1_test" # "test", test is test_java
# "test_cnh_commonvoice"
# "test_vi_commonvoice"
# "test_as_commonvoice"
# "test_pa_IN_commonvoice"
# "test_hi_commonvoice"
# "dev_iban"
# "test_id_commonvoice"
# "test_th_commonvoice"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

./asr.sh \
    --stage 11 \
    --stop_stage 100 \
    --ngpu ${ngpu} \
    --nj 80 \
    --inference_nj 12 \
    --gpu_inference true \
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
    --local_score_opts "--score_lang_id ${lid}"
