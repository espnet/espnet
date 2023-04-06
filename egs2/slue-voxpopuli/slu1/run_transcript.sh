#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel_no_pretrain"
test_sets="devel_no_pretrain"

slu_config=conf/tuning_wavlm/train_asr_conformer_deberta_dropout_old.yaml
inference_config=conf/decode_asr.yaml

./slu.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --use_transcript true\
    --gpu_inference true\
    --stage 11 \
    --stop_stage 13 \
    --token_type bpe \
    --nbpe 1000 \
    --bpe_nlsyms FILL,SEP,PLACE,QUANT,ORG,WHEN,NORP,PERSON,LAW \
    --feats_type raw \
    --audio_format "flac.ark" \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --speed_perturb_factors '0.9 1.0 1.1'\
    --pretrained_model exp/asr_train_asr_conformer_lr5e-4_warmup5k_conv2d_raw_en_bpe1000_sp/valid.acc.ave_10best.pth:encoder:encoder\
    --inference_nj 2 \
    --nj 4 \
    --inference_slu_model valid.acc.ave_10best.pth\
    --slu_config "${slu_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --lm_train_transcript "data/${train_set}/transcript" \
    --bpe_train_text "data/${train_set}/text" "$@"
