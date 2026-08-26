#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# ============================================================================
# HEROICO (LDC2006S37) Spanish ASR recipe.
#
# Corpus: Spanish read + answer speech (heroico) plus the USMA prompt set.
# Pipeline: ESPnet2 asr.sh (Conformer encoder / Transformer decoder, hybrid
#           CTC/attention), character-level tokenization.
#
# Corpus download + Kaldi-style data dirs are handled by local/data.sh
# (stage 0: download into ${HEROICO} from db.sh; stage 1: prepare
# data/{train,dev,test}), invoked at asr.sh stage 1.
# ============================================================================

# ----- Data sets -----------------------------------------------------------
train_set=train
valid_set=dev
test_sets="dev test"

# ----- Configs -------------------------------------------------------------
asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr_transformer.yaml

lm_config=conf/train_lm_transformer.yaml  # kept for future LM experiments; use_lm=false below
use_lm=false
use_wordlm=false

# Speed perturbation is disabled by default. asr.sh applies it in stage 2 and
# then renames the train set to "${train_set}_sp"; leaving it empty keeps the
# train set name as "train". To enable it, set the factors below AND run
# from stage 1 so stage 2 actually builds data/train_sp.
speed_perturb_factors=

./asr.sh \
    --nj 16 \
    --inference_nj 16 \
    --ngpu 1 \
    --lang es \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
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
