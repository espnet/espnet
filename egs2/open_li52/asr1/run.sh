#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

langs="en_de_fr_cy_br_cv_ky_ga-IE_sl_cnh_et_mn_sah_dv_sv-SE_id_ar_ta_ia_lv_ja_rm-sursilv_hsb_ro_fy-NL_el_rm-vallader_as_mt_ka_or_vi_pa-IN_tt_kab_ca_zh-TW_it_fa_eu_es_ru_tr_nl_eo_zh-CN_rw_pt_zh-HK_cs_pl_uk"
lid=true # whether to use language id as additional label

train_set=train_li52_lid
train_dev=dev_li52_lid
test_set="test_ar_commonvoice test_as_commonvoice test_br_commonvoice test_ca_commonvoice test_cnh_commonvoice test_cs_commonvoice test_cv_commonvoice test_cy_commonvoice test_de_commonvoice test_de_voxforge test_dv_commonvoice test_el_commonvoice test_en_commonvoice test_en_voxforge test_eo_commonvoice test_es_commonvoice test_es_voxforge test_et_commonvoice test_eu_commonvoice test_fa_commonvoice test_fr_commonvoice test_fr_voxforge test_fy_NL_commonvoice test_ga_IE_commonvoice test_hsb_commonvoice test_ia_commonvoice test_id_commonvoice test_it_commonvoice test_it_voxforge test_ja_commonvoice test_ka_commonvoice test_kab_commonvoice test_ky_commonvoice test_lv_commonvoice test_mn_commonvoice test_mt_commonvoice test_nl_commonvoice test_nl_voxforge test_or_commonvoice test_pa_IN_commonvoice test_pl_commonvoice test_pt_commonvoice test_pt_voxforge test_rm_sursilv_commonvoice test_rm_vallader_commonvoice test_ro_commonvoice test_ru_commonvoice test_ru_voxforge test_rw_commonvoice test_sah_commonvoice test_sl_commonvoice test_sv_SE_commonvoice test_ta_commonvoice test_tr_commonvoice test_tt_commonvoice test_uk_commonvoice test_vi_commonvoice test_zh_CN_commonvoice test_zh_HK_commonvoice test_zh_TW_commonvoice"

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=4

./asr.sh \
    --lang "${langs}" \
    --local_data_opts "--langs ${langs} --stage 1 --lid ${lid} --nlsyms_txt ${nlsyms_txt}" \
    --stage 10 \
    --stop_stage 10 \
    --nj 40 \
    --ngpu ${ngpu} \
    --use_lm false \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --nlsyms_txt "${nlsyms_txt}" \
    --lm_train_text "data/${train_set}/text" "$@"

