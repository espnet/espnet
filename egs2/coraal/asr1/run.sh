#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


frontend=wavlm
asr_config=conf/tuning/train_asr_${frontend}.yaml
inference_config=conf/decode_asr.yaml
token_type=bpe
nbpe=5000
bpemode=unigram

./asr.sh \
    --lang en \
    --stage 1 \
    --gpu_inference true \
    --ngpu 1 \
    --nbpe "${nbpe}" \
    --use_lm false \
    --max_wav_duration 30 \
    --feats_normalize utt_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.loss.ave.pth" \
    "$@"
