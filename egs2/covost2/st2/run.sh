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
    speed_perturb_factors="0.8 0.9 1.0 1.1 1.2"
else
    speed_perturb_factors="0.9 1.0 1.1"
fi

# Need to take care of real language and the input token tag
kmeans_feature="mhubert_base_vp_en_es_fr_it3/9"  # use model_type/layer_index
nclusters=1000
kmeans_feat_tag=$(echo "${src_lang}_${kmeans_feature}_km${nclusters}" | tr "/" "_")

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
speech_token_case="rm"
src_tgt_text_case="lc.rm"

src_nbpe=3000   # I use src_nbpe=3000 for km1000, and src_nbpe=6000 for km2000.
tgt_nbpe=2000   # 1000 x 2 because there are two languages, if token_joint is True, then only tgt_nbpe is used
if [ ${src_lang} == ja ] || [ ${src_lang} == zh-CN ]; then
    tgt_nbpe=4000
fi

tgt_tasks="asr/mt/st"

st_config=conf/tuning/train_discrete_st_asr_e_branchformer1_km1000_mt0.5.yaml
inference_asr_config=conf/decode_asr.yaml
inference_st_config=conf/decode_st.yaml
inference_mt_config=conf/decode_mt.yaml

train_set=train.${src_lang}-${tgt_lang}
train_dev=dev.${src_lang}-${tgt_lang}
test_sets="test.${src_lang}-${tgt_lang} dev.${src_lang}-${tgt_lang} "


# src_bpe_train_text has multiple es text in order to make it balance
# between speech tokens and es text tokens in the BPE vocabulary due
# to that speech tokens are much longer than text transcriptions.
./st2.sh \
    --kmeans_feature "${kmeans_feature}" \
    --kmeans_opts "--portion 0.5 --nj 2" \
    --nclusters "${nclusters}" \
    --use_lm false \
    --token_joint false \
    --ngpu 2 \
    --nj 16 \
    --inference_nj 16 \
    --speech_token_lang "${kmeans_feat_tag}" \
    --speech_token_case ${speech_token_case} \
    --src_tgt_text_lang "${src_lang}/${tgt_lang}" \
    --src_tgt_text_case ${src_tgt_text_case} \
    --tgt_tasks "${tgt_tasks}" \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --st_config "${st_config}" \
    --inference_asr_config "${inference_asr_config}" \
    --inference_st_config "${inference_st_config}" \
    --inference_mt_config "${inference_mt_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${speech_token_case}.${kmeans_feat_tag} data/${train_set}/text.${src_tgt_text_case}.${src_lang} data/${train_set}/text.${src_tgt_text_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${src_tgt_text_case}.${src_lang} data/${train_set}/text.${src_tgt_text_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${src_tgt_text_case}.${src_lang} data/${train_set}/text.${src_tgt_text_case}.${tgt_lang}" "$@"
