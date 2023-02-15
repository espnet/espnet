#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval2000"

asr_config=conf/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="1.1 0.9 1.0"

./asr.sh \
    --use_lm false \
    --lang en \
    --ngpu 2 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 2 \
    --token_type bpe \
    --nbpe 2000 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_lm valid.loss.best.pth \
    --lm_config "${lm_config}" \
    --score_opts "-s" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" "$@"
