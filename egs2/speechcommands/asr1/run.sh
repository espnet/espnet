#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh

train_set=train_sp
valid_set=dev
test_sets="dev test"

asr_config=conf/tuning/train_asr_transformer_adam.yaml
inference_config=conf/decode_asr.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# lm_train_text is set to avoid errors if speed_perturb is used
./asr.sh                                                \
    --skip_data_prep true                               \
    --skip_train false                                  \
    --skip_eval false                                   \
    --ngpu 1                                            \
    --nj 8                                              \
    --inference_nj 8                                    \
    --speed_perturb_factors "${speed_perturb_factors}"  \
    --feats_type fbank_pitch                            \
    --audio_format wav                                  \
    --fs 16000                                          \
    --token_type word                                   \
    --use_lm false                                      \
    --asr_tag transformer_warmup10k_lr1e-3                     \
    --asr_config "${asr_config}"                        \
    --inference_tag infer                               \
    --inference_config "${inference_config}"            \
    --inference_asr_model valid.loss.ave.pth           \
    --train_set "${train_set}"                          \
    --valid_set "${valid_set}"                          \
    --test_sets "${test_sets}"                          \
    --lm_train_text "data/${train_set}/text" "$@"           
