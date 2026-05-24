#!/usr/bin/env bash
# E-Branchformer (12-block) + Speed Perturbation + Transformer LM rescoring.
#
# Two modes (pick via the WARMSTART env var or --pretrained_model passthrough):
#   WARMSTART=0 (default) -> from scratch, conf/train_asr_e_branchformer.yaml
#   WARMSTART=1           -> warm-start from AISHELL E-Branchformer,
#                            conf/train_asr_e_branchformer_warmstart.yaml,
#                            --pretrained_model + --ignore_init_mismatch true
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

WARMSTART=${WARMSTART:-0}
PRETRAINED_PTH=${PRETRAINED_PTH:-pretrained/aishell_e_branchformer.pth}

if [ "${WARMSTART}" = "1" ]; then
    asr_config=conf/train_asr_e_branchformer_warmstart.yaml
    extra_args=(--pretrained_model "${PRETRAINED_PTH}" --ignore_init_mismatch true)
    asr_tag=e_branchformer_warmstart_aishell
else
    asr_config=conf/train_asr_e_branchformer.yaml
    extra_args=()
    asr_tag=e_branchformer_scratch
fi

inference_config=conf/decode_asr_branchformer_lm.yaml
lm_config=conf/train_lm_transformer.yaml

use_lm=true
use_wordlm=false
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 2 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm} \
    --use_word_lm ${use_wordlm} \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" \
    "${extra_args[@]}" "$@"
