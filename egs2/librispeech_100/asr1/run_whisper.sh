#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
set -x 
eval "$(conda shell.bash hook)"

conda activate /projects/bbjs/slseanwu/ESP_whisper/espnet/tools/anaconda/envs/espnet-whisper

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_tag=conformer_lr4e-5_warmup15k_amp_nondet_whisper_small
asr_config=conf/train_asr_whisper.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 1 \
    --nj 4 \
    --stage 3 \
    --stop_stage 4 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
