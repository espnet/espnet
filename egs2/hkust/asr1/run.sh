#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="dev"


asr_config=conf/tuning/train_asr_transformer2.yaml
inference_config=conf/decode.yaml

lm_config=conf/tuning/train_lm_transformer.yaml
use_lm=true

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu 4 \
    --lang zh                                          \
    --audio_format flac                                \
    --feats_type raw                                   \
    --token_type char                                  \
    --nlsyms_txt data/nlsyms.txt \
    --use_lm ${use_lm}                                 \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
