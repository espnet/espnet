#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=max # Must be "max" for asr. This is to determine how the mixtures are generated in local/data.sh.
sample_rate=16k
n_channels=2


train_set="tr_anechoic_spatialized_${n_chnnels}ch_w_singlespkr"
train_aux_sets="train_si284 tr_anechoic_spatialized_${n_channels}ch"
valid_set="cv_anechoic_spatialized_${n_channels}ch"
test_sets="tt_anechoic_spatialized_${n_channels}ch"

./enh_asr.sh \
    --lang "en" \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --lm_config conf/tuning/train_lm.yaml \
    --joint_config conf/train_asr_transformer.yaml \
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
