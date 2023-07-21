#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train-clean-100"
valid_set="dev"
test_sets="test-clean test-other dev-clean dev-other"

asr_config=conf/tuning/train_asr_e_branchformer_pythia-410m.yaml
inference_config=conf/decode_asr_hf.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 2 \
    --token_type hugging_face \
    --hugging_face_model_name_or_path EleutherAI/pythia-410m-deduped \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
