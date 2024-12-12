#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



test_set="test"
train_set="dev_train"
valid_set="dev_non_train"
nbpe=3436 # 3436 vocabulary size of bpe could cover all the sentence in the edacc dataset
asr_config=conf/train_asr_wavlm_transformer.yaml
# asr_config=conf/train_asr_transformer_finetune.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --nj 16 \
    --inference_nj 2 \
    --gpu_inference true \
    --use_lm false \
    --lang en \
    --ngpu 1 \
    --nbpe "${nbpe}" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --feats_normalize uttmvn \
    --bpe_train_text "data/${train_set}/text" "$@"