#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# Use "--local_data_opts '--tier utt.prono.'" to train on the pronounced
# (colloquial) transcription instead of the orthographic one.  On a corpus this
# small that is worth trying: it drops the pronunciation-to-spelling mapping the
# model otherwise has to learn on top of the acoustics.
local_data_opts=""

# Why these two data settings matter here (see conf/train_asr.yaml for the full
# story): 9.4% of the utterances are a single backchannel syllable, which let the
# first version of this recipe learn the label prior instead of the acoustics, and
# there are only 24 training speakers, so the encoder memorised them.
#   --min_wav_duration 1.0  drops 26% of the utterances but just 5% of the audio,
#                           and asr.sh stage 4 applies it to train/valid only.
# Speed perturbation is done on the fly in conf/train_asr.yaml rather than with
# asr.sh --speed_perturb_factors: that path calls scripts/utils/perturb_data_dir_speed.sh,
# which hard-requires the sox binary (absent here, and tools/installers has no
# sox). The on-the-fly version needs only torchaudio, costs no extra disk, skips a
# re-dump plus re-collect_stats, and redraws the factor every epoch.

./asr.sh \
    --lang ko \
    --nj 32 \
    --inference_nj 8 \
    --audio_format flac.ark \
    --fs 16000 \
    --token_type bpe \
    --nbpe 2000 \
    --use_lm false \
    --min_wav_duration 1.0 \
    --max_wav_duration 20 \
    --local_data_opts "${local_data_opts}" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"
