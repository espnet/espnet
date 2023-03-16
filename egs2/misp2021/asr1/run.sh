#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=train_far
valid_set=dev_far
test_sets=dev_far

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./asr.sh                                   \
    --lang zh \
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 3                               \
    --token_type char                      \
    --feats_type raw                       \
    --use_lm ${use_lm}                     \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --lm_train_text "data/${train_set}/text" "$@"
