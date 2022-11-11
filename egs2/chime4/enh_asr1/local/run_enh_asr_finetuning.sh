#!/usr/bin/env bash
set -e
set -u
set -o pipefail

stage=1
stop_stage=15
train_set=
valid_set=
test_sets=
enh_asr_config=
inference_config=
lm_config=
use_word_lm=
word_vocab_size=
extra_annotations=
ref_channel=
lm_exp=
inference_enh_asr_model=

. ../utils/parse_options.sh

dir=$PWD
cd ../

./enh_asr.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --lang en \
    --spk_num 1 \
    --ref_channel "${ref_channel}" \
    --local_data_opts "--extra-annotations ${extra_annotations}" \
    --nlsyms_txt data/nlsyms.txt \
    --token_type char \
    --feats_type raw \
    --feats_normalize utt_mvn \
    --enh_asr_config "${enh_asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --use_word_lm ${use_word_lm} \
    --word_vocab_size ${word_vocab_size} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text_spk1" \
    --lm_train_text "data/${train_set}/text_spk1 data/local/other_text/text" \
    --lm_exp "${lm_exp}" \
    --inference_enh_asr_model "${inference_enh_asr_model}"

cd $dir
