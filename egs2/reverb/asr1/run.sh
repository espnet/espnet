#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_simu_8ch_si284
valid_set=dt_mult_1ch
test_sets="dt_real_8ch_beamformit dt_simu_8ch_beamformit et_real_8ch_beamformit et_simu_8ch_beamformit dt_real_1ch_wpe dt_simu_1ch_wpe et_real_1ch_wpe et_simu_1ch_wpe dt_real_1ch dt_simu_1ch et_real_1ch et_simu_1ch"

./asr.sh \
    --lang "en" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --use_lm true \
    --token_type char \
    --nbpe 80 \
    --nlsyms_txt data/nlsyms.txt \
    --inference_config conf/decode.yaml \
    --lm_config conf/train_lm_transformer.yaml \
    --asr_config conf/train_asr_transformer.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/tr_simu_8ch_si284/text data/local/other_text/text" "$@"
