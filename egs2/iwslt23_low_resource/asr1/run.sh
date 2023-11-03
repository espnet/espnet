#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_gn
train_dev=dev_gn
test_set=test_gn

asr_config=conf/tuning/train_asr_transformer_totonac.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

nbpe=1000

./asr.sh \
    --lang gn \
    --ngpu 1 \
    --stage 11 \
    --stop_stage 13 \
    --max_wav_duration 40 \
    --gpu_inference true \
    --inference_nj 1 \
    --expdir exp_gn \
    --audio_format "wav" \
    --local_data_opts "--stage 0" \
    --feats_normalize utterance_mvn \
    --inference_asr_model "valid.acc.ave.pth" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type char \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --ignore_init_mismatch true \
    --inference_nj 8 \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"

