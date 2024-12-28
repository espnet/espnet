#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
test_set="dev dev_dialect"

nlsyms_txt=data/local/nlsyms.txt
monolingual_asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --audio_format "wav" \
    --gpu_inference true \
    --inference_nj 64 \
    --use_lm false \
    --inference_asr_model valid.loss.ave.pth \
    --feats_normalize utt_mvn \
    --nlsyms_txt ${nlsyms_txt} \
    --token_type char \
    --feats_type raw \
    --asr_config "${monolingual_asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
