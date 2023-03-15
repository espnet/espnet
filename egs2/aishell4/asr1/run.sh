#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



asr_config=conf/tuning/train_asr_conformer5.yaml
inference_config=conf/tuning/decode_rnn.yaml
lm_config=conf/train_lm_transformer.yaml
use_lm=true
use_wordlm=false


# token_type are char and not bpe for chineese

./asr.sh                                               \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --ngpu 1                                           \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "train_nodev"                       \
    --valid_set "dev"                       \
    --test_sets "test"                        \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --max_wav_duration 20. \
    --lm_train_text "data/train/text" "$@" \
    --nlsyms_txt data/nlsyms.txt
