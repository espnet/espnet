#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="devel_no_pretrain"

slu_config=conf/tuning/train_asr_bert.yaml
inference_config=conf/decode_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5000 \
    --use_transcript true\
    --stage 13 \
    --stop_stage 13 \
    --gpu_inference true \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --speed_perturb_factors '0.9 1.0 1.1'\
    --pretrained_model exp/asr_train_asr_conformer_lr2e-3_warmup5k_conv2d_seed1999_raw_en_word_sp/valid.acc.ave_10best.pth:encoder:encoder\
    --inference_nj 4 \
    --nj 5\
    --inference_asr_model valid.acc.ave_10best.pth\
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"