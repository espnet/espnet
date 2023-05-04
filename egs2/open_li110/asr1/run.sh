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

# train_set=train_pl_voxpopuli
train_set=train_li110_lid
train_dev=dev_li110_lid
test_set="test_ab_commonvoice test_fy_NL_commonvoice test_or_commonvoice \
test_af_openslr32 test_ga_IE_commonvoice test_pa_IN_commonvoice \
test_ar_commonvoice test_gl_commonvoice \
test_as_commonvoice test_gl_openslr77 test_pl_commonvoice \
test_az_commonvoice test_gn_commonvoice test_pl_mls \
test_ba_commonvoice test_gu_openslr78 test_pl_voxpopuli \
test_bas_commonvoice test_ha_commonvoice test_pt_commonvoice \
test_be_commonvoice test_hi_commonvoice test_pt_mls \
test_bg_commonvoice test_hr_voxpopuli test_pt_voxforge \
test_bn_commonvoice test_hsb_commonvoice test_rm_sursilv_commonvoice \
test_bn_openslr37 test_hu_commonvoice \
test_bn_openslr53 test_hu_voxpopuli test_rm_vallader_commonvoice \
test_br_commonvoice test_hy_AM_commonvoice \
test_ca_commonvoice test_ia_commonvoice test_ro_commonvoice \
test_ca_openslr69 test_id_commonvoice test_ro_voxpopuli \
test_ckb_commonvoice test_ig_commonvoice test_ru_commonvoice \
test_cnh_commonvoice test_it_commonvoice test_ru_voxforge \
test_cs_commonvoice test_it_mls test_rw_commonvoice \
test_cs_voxpopuli test_it_voxforge test_sah_commonvoice \
test_cv_commonvoice test_it_voxpopuli test_sat_commonvoice \
test_cy_commonvoice test_ja_commonvoice test_si_openslr52 \
test_da_commonvoice test_jv_openslr35 test_sk_commonvoice \
test_de_commonvoice test_jv_openslr41_female test_sk_voxpopuli \
test_de_mls test_jv_openslr41_male test_sl_commonvoice \
test_de_voxforge test_kab_commonvoice test_sl_voxpopuli \
test_de_voxpopuli test_ka_commonvoice test_sr_commonvoice \
test_dv_commonvoice test_kk_commonvoice test_st_openslr32 \
test_el_commonvoice test_km_openslr42_male test_su_openslr36 \
test_en_commonvoice test_kmr_commonvoice test_su_openslr44_female \
test_en_mls test_su_openslr44_male \
test_en_openslr70 test_kn_openslr79 test_sv_SE_commonvoice \
test_en_voxforge test_ky_commonvoice \
test_en_voxpopuli test_lg_commonvoice test_sw_commonvoice \
test_eo_commonvoice test_lt_commonvoice test_ta_commonvoice \
test_es_commonvoice test_lt_voxpopuli test_ta_openslr65 \
test_es_mls test_lv_commonvoice test_te_openslr66 \
test_es_openslr61 test_mdf_commonvoice test_th_commonvoice \
test_es_openslr71 test_mhr_commonvoice \
test_es_openslr72 test_tn_openslr32 \
test_es_openslr73 test_mk_commonvoice test_tok_commonvoice \
test_es_openslr74 test_ml_commonvoice test_tr_commonvoice \
test_es_openslr75 test_ml_openslr63 test_tt_commonvoice \
test_es_voxforge test_mn_commonvoice test_ug_commonvoice \
test_es_voxpopuli test_mr_commonvoice test_uk_commonvoice \
test_et_commonvoice test_mr_openslr64 test_ur_commonvoice \
test_et_voxpopuli test_mt_commonvoice test_uz_commonvoice \
test_eu_commonvoice test_myv_commonvoice test_vi_commonvoice \
test_eu_openslr76 test_nan_tw_commonvoice test_vot_commonvoice \
test_fa_commonvoice test_ne_openslr43_female test_xh_openslr32 \
test_fi_commonvoice test_ne_openslr54 test_yo_openslr86 \
test_fi_voxpopuli test_nl_commonvoice test_yue_commonvoice \
test_fr_commonvoice test_nl_mls test_zh_CN_commonvoice \
test_fr_mls test_nl_voxforge test_zh_HK_commonvoice \
test_fr_voxforge test_nl_voxpopuli test_zh_TW_commonvoice \
test_fr_voxpopuli test_nn_NO_commonvoice"

test_set="test_ab_commonvoice_lid test_fy_NL_commonvoice_lid test_or_commonvoice_lid \
test_af_openslr32_lid test_ga_IE_commonvoice_lid test_pa_IN_commonvoice_lid \
test_ar_commonvoice_lid test_gl_commonvoice_lid test_as_commonvoice_lid \
test_gl_openslr77_lid test_pl_commonvoice_lid \
test_az_commonvoice_lid test_gn_commonvoice_lid test_pl_mls_lid \
test_ba_commonvoice_lid test_gu_openslr78_lid test_pl_voxpopuli_lid \
test_bas_commonvoice_lid test_ha_commonvoice_lid test_pt_commonvoice_lid \
test_be_commonvoice_lid test_hi_commonvoice_lid test_pt_mls_lid \
test_bg_commonvoice_lid test_hr_voxpopuli_lid test_pt_voxforge_lid \
test_bn_commonvoice_lid test_hsb_commonvoice_lid test_rm_sursilv_commonvoice_lid \
test_bn_openslr37_lid test_hu_commonvoice_lid \
test_bn_openslr53_lid test_hu_voxpopuli_lid test_rm_vallader_commonvoice_lid \
test_br_commonvoice_lid test_hy_AM_commonvoice_lid \
test_ca_commonvoice_lid test_ia_commonvoice_lid test_ro_commonvoice_lid \
test_ca_openslr69_lid test_id_commonvoice_lid test_ro_voxpopuli_lid \
test_ckb_commonvoice_lid test_ig_commonvoice_lid test_ru_commonvoice_lid \
test_cnh_commonvoice_lid test_it_commonvoice_lid test_ru_voxforge_lid \
test_cs_commonvoice_lid test_it_mls_lid test_rw_commonvoice_lid \
test_cs_voxpopuli_lid test_it_voxforge_lid test_sah_commonvoice_lid \
test_cv_commonvoice_lid test_it_voxpopuli_lid test_sat_commonvoice_lid \
test_cy_commonvoice_lid test_ja_commonvoice_lid test_si_openslr52_lid \
test_da_commonvoice_lid test_jv_openslr35_lid test_sk_commonvoice_lid \
test_de_commonvoice_lid test_jv_openslr41_female_lid test_sk_voxpopuli_lid \
test_de_mls_lid test_jv_openslr41_male_lid test_sl_commonvoice_lid \
test_de_voxforge_lid test_kab_commonvoice_lid test_sl_voxpopuli_lid \
test_de_voxpopuli_lid test_ka_commonvoice_lid test_sr_commonvoice_lid \
test_dv_commonvoice_lid test_kk_commonvoice_lid test_st_openslr32_lid \
test_el_commonvoice_lid test_km_openslr42_male_lid test_su_openslr36_lid \
test_en_commonvoice_lid test_kmr_commonvoice_lid test_su_openslr44_female_lid \
test_en_mls_lid test_su_openslr44_male_lid \
test_en_openslr70_lid test_kn_openslr79_lid test_sv_SE_commonvoice_lid \
test_en_voxforge_lid test_ky_commonvoice_lid \
test_en_voxpopuli_lid test_lg_commonvoice_lid test_sw_commonvoice_lid \
test_eo_commonvoice_lid test_lt_commonvoice_lid test_ta_commonvoice_lid \
test_es_commonvoice_lid test_lt_voxpopuli_lid test_ta_openslr65_lid \
test_es_mls_lid test_lv_commonvoice_lid test_te_openslr66_lid \
test_es_openslr61_lid test_mdf_commonvoice_lid test_th_commonvoice_lid \
test_es_openslr71_lid test_mhr_commonvoice_lid \
test_es_openslr72_lid test_tn_openslr32_lid \
test_es_openslr73_lid test_mk_commonvoice_lid test_tok_commonvoice_lid \
test_es_openslr74_lid test_ml_commonvoice_lid test_tr_commonvoice_lid \
test_es_openslr75_lid test_ml_openslr63_lid test_tt_commonvoice_lid \
test_es_voxforge_lid test_mn_commonvoice_lid test_ug_commonvoice_lid \
test_es_voxpopuli_lid test_mr_commonvoice_lid test_uk_commonvoice_lid \
test_et_commonvoice_lid test_mr_openslr64_lid test_ur_commonvoice_lid \
test_et_voxpopuli_lid test_mt_commonvoice_lid test_uz_commonvoice_lid \
test_eu_commonvoice_lid test_myv_commonvoice_lid test_vi_commonvoice_lid \
test_eu_openslr76_lid test_nan_tw_commonvoice_lid test_vot_commonvoice_lid \
test_fa_commonvoice_lid test_ne_openslr43_female_lid test_xh_openslr32_lid \
test_fi_commonvoice_lid test_ne_openslr54_lid test_yo_openslr86_lid \
test_fi_voxpopuli_lid test_nl_commonvoice_lid test_yue_commonvoice_lid \
test_fr_commonvoice_lid test_nl_mls_lid test_zh_CN_commonvoice_lid \
test_fr_mls_lid test_nl_voxforge_lid test_zh_HK_commonvoice_lid \
test_fr_voxforge_lid test_nl_voxpopuli_lid test_zh_TW_commonvoice_lid \
test_fr_voxpopuli_lid test_nn_NO_commonvoice_lid"

nlsyms_txt=data/local/nlsyms.txt
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --local_data_opts "--langs ${langs} --stage 0 --lid ${lid} --nlsyms_txt ${nlsyms_txt}" \
    --ngpu 4 \
    --nj 40 \
    --gpu_inference true \
    --inference_nj 16 \
    --use_lm false \
    --audio_format flac.ark \
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
