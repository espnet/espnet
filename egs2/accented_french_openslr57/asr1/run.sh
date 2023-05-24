#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


asr_config=conf/tuning/train_asr_transformer.yaml

inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm_transformer.yaml

./asr.sh \
    --ngpu 1 \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --inference_nj 10 \
    --nbpe 250 \
    --nj 32 \
    --feats_type raw \
    --inference_asr_model "valid.acc.ave_10best.pth" \
    --feats_normalize "utterance_mvn" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "train" \
    --valid_set "dev" \
    --test_sets "devtest test" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"
