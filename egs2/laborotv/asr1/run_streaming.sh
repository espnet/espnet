#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodev
valid_set=dev_4k
test_sets="dev_4k dev tedx-jp-10k"

asr_config=conf/train_asr_streaming_transformer.yaml
inference_config=conf/decode_asr_streaming.yaml
lm_config=conf/train_lm.yaml
inference_asr_model=valid.acc.ave.pth
# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --use_streaming true \
    --ngpu 4 \
    --nj 128 \
    --inference_nj 256 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@" \
    --inference_asr_model ${inference_asr_model}
