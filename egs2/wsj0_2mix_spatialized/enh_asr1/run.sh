#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


sample_rate=16k     # 8k or 16k
mode="anechoic"     # anechoic or reverb


train_set=tr_spatialized_${mode}_multich_max_${sample_rate}
valid_set=cv_spatialized_${mode}_multich_max_${sample_rate}
test_sets="tt_spatialized_${mode}_multich_max_${sample_rate}"

enh_asr_config=conf/train.yaml
inference_config=conf/decode_asr_transformer.yaml
lm_config=conf/train_lm_transformer.yaml


use_word_lm=false
word_vocab_size=65000

./enh_asr.sh \
    --lang en \
    --audio_format wav \
    --spk_num 2 \
    --ref_channel 3 \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate}" \
    --use_speech_ref true \
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
    --bpe_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2" \
    --lm_train_text "data/${train_set}/text_spk1 data/${train_set}/text_spk2 data/local/other_text/text" "$@"
