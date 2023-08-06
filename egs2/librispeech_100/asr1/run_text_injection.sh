#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

lm_config=conf/train_lm.yaml
asr_config=conf/tuning/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr.yaml

CUDA_VISIBLE_DEVICES="0,1,2,3" ./asr.sh \
    --stage 11 --stop-stage 13 \
    --asr_tag "subword_mean_imbalance_bigger" \
    --post_process_local_data_opts "--stage 5 --stop-stage 6" \
    --post_process_stats_data_opts "--stage 7 --stop-stage 7" \
    --lang en \
    --ngpu 4 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --extra_files "text_injection" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/train_other_500/text data/train_clean_360/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

# CUDA_VISIBLE_DEVICES="0,1,2,3" ./asr.sh \
#     --stage 9 --stop-stage 10 \
#     --asr_tag "subword_fixed_7_imbalance" \
#     --lang en \
#     --ngpu 4 \
#     --nj 16 \
#     --gpu_inference true \
#     --inference_nj 2 \
#     --nbpe 5000 \
#     --max_wav_duration 30 \
#     --speed_perturb_factors "0.9 1.0 1.1" \
#     --audio_format "flac.ark" \
#     --feats_type raw \
#     --use_lm false \
#     --asr_config "${asr_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --extra_files "text_injection" \
#     --test_sets "${test_sets}" \
#     --lm_train_text "data/${train_set}/text" \
#     --bpe_train_text "data/${train_set}/text" "$@"
