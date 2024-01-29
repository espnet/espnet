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
asr_config=conf/tuning/train_asr_e_branchformer_size512_mlp3072_linear1024_e17_mactrue_edrop0.0_ddrop0.0_text_injection.yaml
inference_config=conf/decode_asr.yaml

text_injection_token_types="bpe char phn"

./asr.sh \
    --stage 11 --stop-stage 13 \
    --asr_tag "subword_fixed_3_bpe_char_phn_test" \
    --post_process_local_data_opts "--stage 5 --stop-stage 6" \
    --post_process_stats_data_opts "--stage 7 --stop-stage 7" \
    --lang en \
    --ngpu 1 \
    --nj 32  \
    --gpu_inference true \
    --inference_nj 2 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --extra_files "text_injection" \
    --g2p "g2p_en" \
    --text_injection_token_types "${text_injection_token_types}" \
    --text_injection_extra_files "text_injection" \
    --lm_train_text "data/${train_set}/text data/train_other_500/text data/train_clean_360/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
