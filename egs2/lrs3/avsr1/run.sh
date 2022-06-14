#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"


asr_tag=train_avsr_transformer_audio_only
asr_config=conf/train_avsr_transformer.yaml
lm_config=conf/train_lm.yaml  # Not Used, as use_lm=false

export CUDA_VISIBLE_DEVICES=3

./avsr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --stage 1 \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 16 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --lm_config ${lm_config} \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --multimodal true \
    --audio_input true \
    --vision_input true \
    --mouth_roi true \
    --audio_sample_step 1 \
    --vision_sample_step 10 \
    --stack_order 2 \
    --avg_pool_width 1 \
    --align_option "duplicate" \
    --fusion_stage "frontend" \
    --fusion_type "concat" \
