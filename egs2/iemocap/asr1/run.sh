#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
local_data_opts="--lowercase true --remove_punctuation true --remove_emo xxx_exc_fru_fea_sur"
# local_data_opts: 5 following options can be set (default=false)
#--lowercase
#   Convert transcripts into lowercase if "true".
#--remove_punctuation
#   Remove punctuation (except apostrophes) if "true".
#--remove_tag
#   Remove [TAGS] (e.g.[LAUGHTER]) if "true".
#--remove_emo
#   Remove the utterances with the specified emotional labels.
#   If specifying two or more labels, concatenate them with "_" (e.g. xxx_exc_fru_fea_sur).
#   emotional labels: ang (anger), hap (happiness), exc (excitement), sad (sadness),
#   fru (frustration), fea (fear), sur (surprise), neu (neutral), and xxx (other)
#--convert_to_sentiment
#   Convert emotion to sentiment (Positive, Negative and Neutral)
#   mapping from emotion to sentiment is as follows:
#   Positive: hap, exc, sur
#   Negative: ang, sad, fru, fea
#   Neutral: neu
#   This option normalizes text irrelevant of "--lowercase" "--remove_punctuation" "--remove_tag" options

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "${local_data_opts}" "$@"
