#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_tag=conformer_lr1e-3_warmup25k

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr.yaml
inference_asr_model=valid.acc.ave.pth

use_lm=false
use_wordlm=false
lm_config=conf/train_lm_transformer.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                                \
    --skip_data_prep false                              \
    --skip_train false                                  \
    --skip_eval false                                   \
    --lang zh                                           \
    --audio_format wav                                  \
    --feats_type raw                                    \
    --token_type char                                   \
    --ngpu 4                                            \
    --asr_tag "${asr_tag}"                              \
    --use_lm "${use_lm}"                                \
    --use_word_lm "${use_wordlm}"                       \
    --lm_config "${lm_config}"                          \
    --asr_config "${asr_config}"                        \
    --inference_config "${inference_config}"            \
    --inference_asr_model "${inference_asr_model}"      \
    --train_set "${train_set}"                          \
    --valid_set "${valid_set}"                          \
    --test_sets "${test_sets}"                          \
    --speed_perturb_factors "${speed_perturb_factors}"  \
    --lm_train_text "data/${train_set}/text" "$@"
