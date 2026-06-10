#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_nocut"
valid_set="dev_nocut"
test_sets="test"

# Set this to one of ["legacy", "speaker-adaptation"]
data_type=legacy

asr_config=conf/train.yaml
inference_config=conf/decode.yaml
lm_config=conf/train_lm_transformer.yaml

./asr.sh \
    --lang en \
    --nj 8 \
    --expdir exp_long \
    --ngpu 1 \
    --gpu_inference true \
    --inference_nj 1 \
    --feats_type raw \
    --audio_format "wav" \
    --token_type bpe \
    --nbpe 500 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/local/text" \
    --lm_config "${lm_config}" \
    --local_data_opts "--data_type ${data_type}" \
    --bpe_nlsyms "[unk]" "$@"
