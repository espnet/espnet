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

