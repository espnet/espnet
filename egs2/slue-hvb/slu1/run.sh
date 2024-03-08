#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

local_data_opts="--use_classifier false"
asr_config=conf/tuning/wavlm_complex.yaml
inference_config=conf/tuning/decode_asr_maxlenratio_80_inf.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --speed_perturb_factors "1.1 0.9 1.0" \
    --lm_train_text "dump/raw/train_sp/text" \
    --nj 1\
    --use_lm false \
    --nbpe 5000 \
    --gpu_inference true \
    --feats_normalize utterance_mvn\
    --inference_nj 4 \
    --bpe_nlsyms "[unk]" \
    --token_type word\
    --audio_format flac\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_slu_model valid.acc.ave_10best.pth\
    --slu_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --local_data_opts "${local_data_opts}" \
    --test_sets "${test_sets}" "$@"
