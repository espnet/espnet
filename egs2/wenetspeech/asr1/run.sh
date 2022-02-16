#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

set=L    # S for the small set, M for the mediate set, L for the large set

train_set=train_"$(echo "${set}" | tr "[:lower:]" "[:upper:]")"
valid_set=dev
test_sets="dev test_meeting test_net"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

lm_config=conf/train_lm.yaml
use_lm=false

# speed perturbation related
# add "--speed_perturb_factors="0.9 1.0 1.1" if you want to
# apply speed perturbation for the training data

./asr.sh                                               \
    --lang zh                                          \
    --local_data_opts "--set ${set}"                   \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
