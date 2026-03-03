#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# ===== Configuration =====
lang=en
gender=male  # "male" or "female"

train_set="train_${gender}_${lang}"
train_dev="dev_${gender}_${lang}"
test_set="${train_dev} test_${gender}_${lang}"

asr_config=conf/tuning/train_asr_conformer5.yaml
inference_config=conf/decode_asr.yaml

nbpe=150

./asr.sh \
    --nj 8 \
    --inference_nj 8 \
    --ngpu 1 \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang} --gender ${gender}" \
    --use_lm false \
    --token_type bpe \
    --nbpe "${nbpe}" \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
