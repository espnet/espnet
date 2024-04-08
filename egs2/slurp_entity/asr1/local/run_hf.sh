#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_char"
valid_set="devel_char"
test_sets="test_char devel_char"

asr_config=conf/tuning/train_asr_branchformer_xlsr_mbart.yaml
inference_config=conf/decode_asr_hf.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type hugging_face \
    --hugging_face_model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --local_data_opts "--token_type_bpe false" \
    --local_score_opts "--token_type_bpe false" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_nj 1 \
    --gpu_inference true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --lm_train_text "data/${train_set}/text" \
    --test_sets "${test_sets}" "$@"
