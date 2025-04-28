#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=valid
test_sets="test"

slu_config=conf/train_asr_whisper_turn_taking.yaml
inference_config=conf/decode_asr_chunk.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
# speed_perturb_factors="1.1 0.9 1.0"

./slu.sh \
    --use_lm false \
    --lang en \
    --ngpu 2 \
    --nj 32 \
    --gpu_inference true \
    --inference_nj 4 \
    --inference_slu_model valid.loss.ave.pth \
    --token_type word \
    --nbpe 2000 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" \
    --slu_config "${slu_config}" \
    --inference_config "${inference_config}" \
    --inference_lm valid.loss.best.pth \
    --lm_config "${lm_config}" \
    --score_opts "-s" \
    --feats_normalize utterance_mvn\
    --max_wav_duration 40\
    --no_asr_eval true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
