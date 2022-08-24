#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="marathi_train"
train_dev="marathi_dev"
test_set="marathi_test"

asr_config=conf/tuning/train_asr_conformer_xlsr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

./asr.sh \
    --stage 1 \
    --stop_stage 100 \
    --ngpu ${ngpu} \
    --nj 10 \
    --inference_nj 10 \
    --gpu_inference true \
    --audio_format "wav" \
    --inference_args "--batch_size 1" \
    --use_lm false \
    --token_type bpe \
    --nbpe 150 \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"
