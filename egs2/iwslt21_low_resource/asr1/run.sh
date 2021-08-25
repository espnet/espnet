#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_babel_alffa_gamayun_iwslt"
valid_set="valid_alffa_iwslt"
test_sets="test_iwslt_swa test_iwslt_swc test_iwslt_swa_raw test_iwslt_swc_raw"

asr_config=conf/train_asr_conformer.yaml
lm_config=conf/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang sw \
    --ngpu 2 \
    --nbpe 100 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model valid.acc.ave.pth \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --lm_test_text "data/test_iwslt/text" \
    --bpe_train_text "data/${train_set}/text" \
    --bpe_nlsyms "$(perl -pe 's/\n/,/' local/nlsyms.txt)" \
    --nlsyms_txt local/nlsyms.txt "$@"
