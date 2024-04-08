#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

asr_config=conf/tuning/train_asr_branchformer_wavlm_mbart.yaml
inference_config=conf/decode_asr_hf.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type hugging_face \
    --hugging_face_model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --local_score_opts "--score_folder score_wer" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --lm_train_text "data/${train_set}/text" \
    --test_sets "${test_sets}" "$@"
