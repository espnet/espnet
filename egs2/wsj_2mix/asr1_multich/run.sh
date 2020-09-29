#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=max # Must be "max" for asr. This is to determine how the mixtures are generated in local/data.sh.
sample_rate=16k


train_set="tr_singlespkr_anechoic_spatialized_2ch"  # "tr_anechoic_spatialized_2ch" #"tr_${min_or_max}_${sample_rate}"
train_aux_sets="train_si284_1 tr_anechoic_spatialized_2ch"
valid_set="cv_anechoic_spatialized_2ch" #"cv_${min_or_max}_${sample_rate}"
test_sets="tt_anechoic_spatialized_2ch_32utts" #"tt_${min_or_max}_${sample_rate}"

./enh_asr.sh \
    --lang "en" \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/tuning/train_asr_transformer_singlespkr_5.yaml \
    --train_set "${train_set}" \
    --train_aux_sets "${train_aux_sets}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --use_signal_ref false \
    --max_wav_duration 2000 \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --srctexts "data/wsj/train_si284/text data/wsj/local/other_text/text" "$@"
