#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# language related
src_lang=es
tgt_lang=en
# English (en)
# French (fr)
# German (de)
# Spanish (es)
# Catalan (ca)
# Italian (it)
# Russian (ru)
# Chinese (zh-CN)
# Portuguese (pt)
# Persian (fa)
# Estonian (et)
# Mongolian (mn)
# Dutch (nl)
# Turkish (tr)
# Arabic (ar)
# Swedish (sv-SE)
# Latvian (lv)
# Slovenian (sl)
# Tamil (ta)
# Japanese (ja)
# Indonesian (id)
# Welsh (cy)
nbpe=1000

train_set=train.${src_lang}-${tgt_lang}
train_dev=dev.${src_lang}-${tgt_lang}
test_set="test.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang} "

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# verify language directions
is_exist=false
is_low_resource=false
if [[ ${src_lang} == en ]]; then
    tgt_langs=de_ca_zh-CN_fa_et_mn_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${tgt_langs} | tr '_' ' '); do
        if [[ ${lang} == "${tgt_lang}" ]]; then
            is_exist=true
            break
        fi
    done
else
    lr_src_langs=it_ru_zh-CN_pt_fa_et_mn_nl_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${lr_src_langs} | tr '_' ' '); do
        if [[ ${lang} == "${src_lang}" ]]; then
            is_low_resource=true
            break
        fi
    done
    src_langs=fr_de_es_ca_it_ru_zh-CN_pt_fa_et_mn_nl_tr_ar_sv-SE_lv_sl_ta_ja_id_cy
    for lang in $(echo ${src_langs} | tr '_' ' '); do
        if [[ ${lang} == "${src_lang}" ]]; then
            is_exist=true
            break
        fi
    done
fi
if [[ ${is_exist} == false ]]; then
    echo "No language direction: ${src_lang} to ${tgt_lang}" && exit 1;
fi

if [ ${is_low_resource} = true ]; then
    speed_perturb_factors="0.9 1.0 1.1"
else
    speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"
fi

if [ ${src_lang} == ja ] || [ ${src_lang} == zh-CN ]; then
    nbpe=4000
fi

./asr.sh \
    --ngpu 1 \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --tgt_lang ${tgt_lang}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
