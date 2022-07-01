#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

if python3 -c 'import torch as t; from packaging.version import parse as L; assert L(t.__version__) >= L("1.7.0")' &> /dev/null;  then
	asr_config=conf/train_asr.yaml
else
	asr_config=conf/tuning/train_asr_transformer_adam_specaug.yaml #s3prl is installed when pytorch > 1.7. Hence using default frontend
fi

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
