#!/usr/bin/env bash
# SOT multi-talker ASR recipe for AMI dataset using Whisper.
#
# Follows the same pattern as egs2/librimix/sot_asr1 but uses
# Whisper encoder/decoder with timestamps and cpWER evaluation.
#
# Prerequisites:
#   - Kaldi-format data directories in data/{train,dev,test}/
#     (use local/prepare_sot.py to create from Lhotse CutSets)
#
# Usage:
#   ./run.sh --stage 1 --stop_stage 13
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/tuning/train_sot_asr_whisper_small.yaml
inference_config=conf/tuning/decode_sot.yaml

./asr.sh \
    --lang en \
    --feats_type raw \
    --token_type whisper_multilingual \
    --sot_asr false \
    --max_wav_duration 30 \
    --feats_normalize null \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
