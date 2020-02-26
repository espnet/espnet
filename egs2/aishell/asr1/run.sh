#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_sets="test "

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode_asr_rnn.yaml

lm_config=conf/train_lm.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --audio_format wav                                 \
    --feats_type fbank_pitch                           \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --decode_config "${decode_config}"                 \
    --train_set "${train_set}"                         \
    --dev_set "${dev_set}"                             \
    --eval_sets "${eval_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --srctexts "data/${train_set}/text" "$@"
