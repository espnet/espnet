#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

langs="ar_as_br_ca_cnh_cs_cv_cy_de_dv\
_el_en_eo_es_et_eu_fa_fr_fy-NL_ga-IE_hsb\
_ia_id_it_ja_ka_kab_ky_lv_mn_mt_nl_or\
_pa-IN_pl_pt_rm-sursilv_rm-vallader_ro\
_ru_rw_sah_sl_sv-SE_ta_tr_tt_uk_vi_\
zh-CN_zh-HK_zh-TW"
lid=true # whether to use language id as additional label

train_set=train_li52_lid
train_dev=dev_li52_lid
# high_resource (>100h): ca, de, en, es, fa, fr, kab, it, rw, ru, pl (11)
high_resource_test_set="test_ca_commonvoice_lid test_de_commonvoice_lid \
test_de_voxforge_lid test_en_commonvoice_lid test_en_voxforge_lid \
test_es_commonvoice_lid test_es_voxforge_lid test_fa_commonvoice_lid \
test_fr_commonvoice_lid test_fr_voxforge_lid test_kab_commonvoice_lid \
test_it_commonvoice_lid test_it_voxforge_lid test_rw_commonvoice_lid \
test_ru_commonvoice_lid test_ru_voxforge_lid test_pl_commonvoice_lid"
# mid_resource (20-100h): cs, cy, eo, eu, nl, pt, tr, tt, uk, zh_CN, zh_HK, zh_TW (12)
mid_resource_test_set="test_cs_commonvoice_lid test_cy_commonvoice_lid \
test_eo_commonvoice_lid test_eu_commonvoice_lid test_nl_voxforge_lid \
test_pt_commonvoice_lid test_tr_commonvoice_lid test_tt_commonvoice_lid \
test_uk_commonvoice_lid test_zh_CN_commonvoice_lid test_zh_HK_commonvoice_lid \
test_zh_TW_commonvoice_lid"
# low_resource (5-20h): ar, br, dv, el, et, fy_NL, id, ja, ky, mn, mt, sv, ta (13)
# we use this set as a default test set
low_resource_test_set="test_ar_commonvoice_lid test_br_commonvoice_lid \
test_dv_commonvoice_lid test_el_commonvoice_lid test_et_commonvoice_lid \
test_fy_NL_commonvoice_lid test_id_commonvoice_lid test_ja_commonvoice_lid \
test_ky_commonvoice_lid test_mn_commonvoice_lid test_mt_commonvoice_lid \
test_sv_SE_commonvoice_lid test_ta_commonvoice_lid"
# extremely low resource (<=5h): as, cnh, cv, ga_IE, hsb, ia, ka, lv, or, pa_IN,
#                                rm_sursilv, rm_vallader, ro, sah, sl, vi (16)
ext_low_resource_test_set="test_as_commonvoice_lid test_cnh_commonvoice_lid \
test_cv_commonvoice_lid test_ga_IE_commonvoice_lid test_hsb_commonvoice_lid \
test_ia_commonvoice_lid test_ka_commonvoice_lid test_lv_commonvoice_lid \
test_or_commonvoice_lid test_pa_IN_commonvoice_lid test_rm_sursilv_commonvoice_lid \
test_rm_vallader_commonvoice_lid test_ro_commonvoice_lid test_sah_commonvoice_lid \
test_sl_commonvoice_lid test_vi_commonvoice_lid"

full_set="${high_resource_test_set} ${mid_resource_test_set} ${low_resource_test_set} ${ext_low_resource_test_set}"
test_set=${full_set}
# use the middle resource test set to avoid too long evaluation time
test_set=${mid_resource_test_set}

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --local_data_opts "--langs ${langs} --stage 0 --lid ${lid} --nlsyms_txt ${nlsyms_txt}" \
    --stage 1 \
    --stop_stage 100 \
    --ngpu 4 \
    --nj 80 \
    --inference_nj 256 \
    --use_lm false \
    --token_type bpe \
    --nbpe 7000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --nlsyms_txt "${nlsyms_txt}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"
