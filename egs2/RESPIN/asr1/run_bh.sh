#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
lang="Bhojpuri"
train_set=$lang/"train"
valid_set=$lang/"dev"
test_sets=$lang/"test_unseen_spk_sent"

asr_config=conf/tuning/train_asr_conformer6_n_fft400_hop_length160_multiple_encoders.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml


./asr.sh \
    --lang $lang \
    --ngpu 2 \
    --stage 1 \
    --token_type "char" \
    --max_wav_duration 15 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --nj 64 \
	

