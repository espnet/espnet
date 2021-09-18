#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr_rnn.yaml
lm_config=conf/train_lm_transformer.yaml
use_lm=true
use_wordlm=false

#pour l'instant je vais mettre les 3 train pour train/dev/test
# token_type are char and not bpe for chineese ! (and Japaneese)

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
    --test_sets "dev"                        \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --max_wav_duration 20. \
    --lm_train_text "data/train/text" "$@" \
    --nlsyms_txt /ocean/projects/cis210027p/berrebbi/espnet/egs2/aishell4/asr1/data/nlsyms.txt