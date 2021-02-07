# dirs="dev_tr_commonvoice dev_tt_commonvoice dev_uk_commonvoice dev_vi_commonvoice dev_zh_CN_commonvoice dev_zh_HK_commonvoice dev_zh_TW_commonvoice"
dirs="train_de_voxforge train_es_voxforge train_fr_voxforge train_it_voxforge train_nl_voxforge train_pt_voxforge train_ru_voxforge"
# dirs="test_en_commonvoice test_en_voxforge test_eo_commonvoice test_es_commonvoice test_es_voxforge test_et_commonvoice test_eu_commonvoice test_fa_commonvoice test_fr_commonvoice test_fr_voxforge test_fy_NL_commonvoice test_ga_IE_commonvoice test_hsb_commonvoice test_ia_commonvoice test_id_commonvoice test_it_commonvoice test_it_voxforge test_ja_commonvoice test_ka_commonvoice test_kab_commonvoice test_ky_commonvoice test_lv_commonvoice test_mn_commonvoice test_mt_commonvoice test_nl_commonvoice test_nl_voxforge test_or_commonvoice test_pa_IN_commonvoice test_pl_commonvoice test_pt_commonvoice test_pt_voxforge test_rm_sursilv_commonvoice test_rm_vallader_commonvoice test_ro_commonvoice test_ru_commonvoice test_ru_voxforge test_rw_commonvoice test_sah_commonvoice test_sl_commonvoice test_sv_SE_commonvoice test_ta_commonvoice test_tr_commonvoice test_tt_commonvoice test_uk_commonvoice test_vi_commonvoice test_zh_CN_commonvoice test_zh_HK_commonvoice test_zh_TW_commonvoice"
# for x in data/test_* ; do
for x in ${dirs}; do
  # utils/fix_data_dir.sh data/${x}
  echo "${x}"
  utils/data/get_utt2dur.sh data/${x}
  awk 'BEGIN{SUM=0}{SUM+=$2}END{print SUM/3600}' data/${x}/utt2dur
done

