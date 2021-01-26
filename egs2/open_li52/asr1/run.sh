#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

langs="ar_as_br_ca_cnh_cs_cv_cy_de_dv\
_el_eo_es_et_eu_fa_fr_fy-NL_ga-IE_hsb\
_ia_id_it_ja_ka_kab_ky_lv_mn_mt_nl_or\
_pa-IN_pl_pt_rm-sursilv_rm-vallader_ro\
_ru_rw_sah_sl_sv-SE_ta_tr_tt_uk_vi_\
zh-CN_zh-HK_zh-TW"
lid=true # whether to use language id as additional label

train_set=train_li52_lid
train_dev=dev_li52_lid
# test_set="test_ar_commonvoice test_as_commonvoice\
#  test_br_commonvoice test_ca_commonvoice test_cnh_commonvoice\
#  test_cs_commonvoice test_cv_commonvoice test_cy_commonvoice\
#  test_de_commonvoice test_de_voxforge test_dv_commonvoice\
#  test_el_commonvoice test_en_commonvoice test_en_voxforge\
#  test_eo_commonvoice test_es_commonvoice test_es_voxforge\
#  test_et_commonvoice test_eu_commonvoice test_fa_commonvoice\
#  test_fr_commonvoice test_fr_voxforge test_fy_NL_commonvoice\
#  test_ga_IE_commonvoice test_hsb_commonvoice test_ia_commonvoice\
#  test_id_commonvoice test_it_commonvoice test_it_voxforge\
#  test_ja_commonvoice test_ka_commonvoice test_kab_commonvoice\
#  test_ky_commonvoice test_lv_commonvoice test_mn_commonvoice\
#  test_mt_commonvoice test_nl_commonvoice test_nl_voxforge\
#  test_or_commonvoice test_pa_IN_commonvoice test_pl_commonvoice\
#  test_pt_commonvoice test_pt_voxforge test_rm_sursilv_commonvoice\
#  test_rm_vallader_commonvoice test_ro_commonvoice test_ru_commonvoice\
#  test_ru_voxforge test_rw_commonvoice test_sah_commonvoice\
#  test_sl_commonvoice test_sv_SE_commonvoice test_ta_commonvoice\
#  test_tr_commonvoice test_tt_commonvoice test_uk_commonvoice\
#  test_vi_commonvoice test_zh_CN_commonvoice test_zh_HK_commonvoice\
#  test_zh_TW_commonvoice"

test_set="test_ar_commonvoice_lid test_as_commonvoice_lid test_br_commonvoice_lid\ 
test_ca_commonvoice_lid test_cnh_commonvoice_lid test_cs_commonvoice_lid\ 
test_cv_commonvoice_lid test_cy_commonvoice_lid test_de_commonvoice_lid\ 
test_de_voxforge_lid test_dv_commonvoice_lid test_el_commonvoice_lid\ 
test_en_commonvoice_lid test_en_voxforge_lid test_eo_commonvoice_lid\ 
test_es_commonvoice_lid test_es_voxforge_lid test_et_commonvoice_lid\ 
test_eu_commonvoice_lid test_fa_commonvoice_lid test_fr_commonvoice_lid\ 
test_fr_voxforge_lid test_fy_NL_commonvoice_lid test_ga_IE_commonvoice_lid\ 
test_hsb_commonvoice_lid test_ia_commonvoice_lid test_id_commonvoice_lid\ 
test_it_commonvoice_lid test_it_voxforge_lid test_ja_commonvoice_lid\ 
test_ka_commonvoice_lid test_kab_commonvoice_lid test_ky_commonvoice_lid\ 
test_lv_commonvoice_lid test_mn_commonvoice_lid test_mt_commonvoice_lid\ 
test_nl_commonvoice_lid test_nl_voxforge_lid test_or_commonvoice_lid\ 
test_pa_IN_commonvoice_lid test_pl_commonvoice_lid test_pt_commonvoice_lid\ 
test_pt_voxforge_lid test_rm_sursilv_commonvoice_lid test_rm_vallader_commonvoice_lid\ 
test_ro_commonvoice_lid test_ru_commonvoice_lid test_ru_voxforge_lid\ 
test_rw_commonvoice_lid test_sah_commonvoice_lid test_sl_commonvoice_lid\ 
test_sv_SE_commonvoice_lid test_ta_commonvoice_lid test_tr_commonvoice_lid\ 
test_tt_commonvoice_lid test_uk_commonvoice_lid test_vi_commonvoice_lid\ 
test_zh_CN_commonvoice_lid test_zh_HK_commonvoice_lid test_zh_TW_commonvoice_lid"

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

./asr.sh \
    --local_data_opts "--langs ${langs} --stage 0 --lid ${lid} --nlsyms_txt ${nlsyms_txt}" \
    --stage 1 \
    --stop_stage 100 \
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

