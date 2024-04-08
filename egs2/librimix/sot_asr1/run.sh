#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/tuning/train_sot_asr_conformer_wavlm.yaml

lm_config=conf/tuning/train_lm_transformer.yaml
inference_config=conf/tuning/decode_sot.yaml

./asr.sh \
    --lang en \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --sot_asr true \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_normalize utterance_mvn \
    --use_lm false \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2 data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2" "$@"
