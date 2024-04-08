#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_tag=conformer_mono16k_warmup800_lr2e-4_accum2
inference_tag=infer

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 0.95 1.0 1.05 1.1"

./asr.sh                                                \
    --skip_data_prep false                              \
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
    --asr_tag "${asr_tag}"                              \
    --asr_config "${asr_config}"                        \
    --inference_tag "${inference_tag}"                  \
    --inference_config "${inference_config}"            \
    --inference_asr_model valid.acc.ave.pth             \
    --train_set "${train_set}"                          \
    --valid_set "${valid_set}"                          \
    --test_sets "${test_sets}"                          \
    --local_score_opts "--inference_tag ${inference_tag}" \
    --lm_train_text "data/${train_set}/text" "$@"
