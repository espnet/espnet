#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

module load cuda/10.2.0 cudnn mkl

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/train_asr_wav2vec2_xlsr_with_valid.yaml
inference_config=conf/decode_asr.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --ngpu 1 \
    --stage 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --feats_normalize null \
    --max_wav_duration 30 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --expdir "exp_$(basename ${asr_config%.*})" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
