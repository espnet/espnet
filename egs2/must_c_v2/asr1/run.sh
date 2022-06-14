#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=en
tgt_lang=de

train_set=train.en-${tgt_lang}
train_dev=dev.en-${tgt_lang}
test_set="tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}"

asr_config=conf/tuning/train_asr_conformer.yaml
inference_config=conf/tuning/decode_asr_conformer.yaml

nbpe=4000

./asr.sh \
    --use_lm false \
    --local_data_opts "${tgt_lang}" \
    --audio_format "flac.ark" \
    --nj 40 \
    --inference_nj 40 \
    --audio_format "flac.ark" \
    --token_type "bpe" \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text"  "$@"
