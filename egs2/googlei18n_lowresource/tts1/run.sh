#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 44100 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

lang="es_ar"
openslr_id=61
sex=both

# lang_id: lang full name (openslr_id) female/male/both
# es_ar: Argentinian Spanish (61) both
# ml_in: Malayalam (63) both
# mr_in: Marathi (64) female
# ta_in: Tamil (65) both
# te_in: Telugu (66) both
# ca_es: Catalan (69) both
# en_ng: Nigerian (70) both
# es_cl: Chilean (71) both
# es_co: Colombian (72) both
# es_pe: Peruvian (73) both
# es_pr: Puerto Rico Spanish (74) female
# es_ve: Venezuelan Spanish (75) both
# eu_es: Basque (76) both
# gl_es: Galician (77) both
# gu_in: Gujarati (78) both
# kn_in: Kannada (79) both
# my_mm: Burmese (80) female
# irish_english: Ireland English (83) male
# midlands_english: Midland English (83) both
# northern_english: Northern English (83) both
# scottish_english: Scottish English (83) both
# southern_english: Southern English (83) both
# welsh_english: Welsh English (83) both
# yo_ng: Yoruba (86) both

train_config=conf/train.yaml
inference_config=conf/decode.yaml

train_set=train_no_dev_${lang}
valid_set=dev_${lang}
test_sets="dev_${lang} test_${lang}"

# no g2p for crowdsourced languages
g2p=none

./tts.sh \
    --lang ${lang} \
    --local_data_opts "--lang ${lang} --openslr_id ${openslr_id} --sex ${sex}" \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type raw \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --use_xvector true \
    ${opts} "$@"
