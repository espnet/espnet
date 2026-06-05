#!/usr/bin/env bash
# Default run for MagicData-RAMC ASR: E-Branchformer-12 + speed perturbation
# + Transformer LM rescoring (the recipe README's recommended row 3b).
# Variants are reached by overriding any of these defaults at the CLI; see
# README.md for the canonical override snippets.
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_e_branchformer.yaml
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
    --nlsyms_txt data/nlsyms.txt                       \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
