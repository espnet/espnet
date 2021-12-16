# RESULTS
## Environments
- date: `Fri Feb 12 03:20:12 EST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.7`
- pytorch version: `pytorch 1.7.1`
- Git hash: `fcf20cc33acd09b8ae80c88703aee0b755ed7652`
  - Commit date: `Tue Feb 9 22:22:58 2021 -0500`

## asr_train_asr_transformer_e45_raw_bpe7000
- model link: https://zenodo.org/record/4509663

### CER (we only list mid_resource_test_set)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_cs_commonvoice_lid|2574|100007|79.7|14.3|6.0|4.4|24.7|97.2|
|decode_asr_asr_model_valid.acc.ave/test_cy_commonvoice_lid|2937|139576|87.9|7.0|5.2|3.3|15.4|87.8|
|decode_asr_asr_model_valid.acc.ave/test_eo_commonvoice_lid|8453|394733|89.4|7.2|3.4|3.4|14.0|90.3|
|decode_asr_asr_model_valid.acc.ave/test_eu_commonvoice_lid|4912|274944|93.9|3.2|2.9|1.8|7.9|89.6|
|decode_asr_asr_model_valid.acc.ave/test_nl_voxforge_lid|874|46666|88.4|6.1|5.5|3.3|14.9|97.3|
|decode_asr_asr_model_valid.acc.ave/test_pt_commonvoice_lid|4334|191499|81.8|10.6|7.7|3.8|22.0|95.7|
|decode_asr_asr_model_valid.acc.ave/test_tr_commonvoice_lid|1639|67841|79.3|13.0|7.7|3.1|23.8|96.6|
|decode_asr_asr_model_valid.acc.ave/test_tt_commonvoice_lid|4365|159202|61.5|31.9|6.7|4.3|42.8|99.2|
|decode_asr_asr_model_valid.acc.ave/test_uk_commonvoice_lid|1671|80323|79.0|14.9|6.1|3.4|24.5|98.9|
|decode_asr_asr_model_valid.acc.ave/test_zh_CN_commonvoice_lid|8273|134476|67.2|30.6|2.3|4.1|36.9|96.6|
|decode_asr_asr_model_valid.acc.ave/test_zh_HK_commonvoice_lid|2805|32994|85.6|13.3|1.1|2.5|16.9|62.1|
|decode_asr_asr_model_valid.acc.ave/test_zh_TW_commonvoice_lid|2627|21991|80.0|18.5|1.5|5.5|25.6|63.4|

## asr_train_asr_transformer_e45_raw_bpe8000
- model link: https://zenodo.org/record/4509671
### CER (we only list mid_resource_test_set)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_cs_commonvoice_lid|2574|100007|78.3|15.8|5.9|4.9|26.5|97.3|
|decode_asr_asr_model_valid.acc.ave/test_cy_commonvoice_lid|2937|139576|88.9|6.4|4.7|3.3|14.4|84.7|
|decode_asr_asr_model_valid.acc.ave/test_eo_commonvoice_lid|8453|394733|90.2|6.5|3.2|3.2|13.0|89.1|
|decode_asr_asr_model_valid.acc.ave/test_eu_commonvoice_lid|4912|274944|94.0|3.1|2.9|1.7|7.7|88.2|
|decode_asr_asr_model_valid.acc.ave/test_nl_voxforge_lid|874|46666|88.6|5.9|5.5|3.2|14.6|96.2|
|decode_asr_asr_model_valid.acc.ave/test_pt_commonvoice_lid|4334|191499|81.7|10.9|7.5|3.8|22.2|95.4|
|decode_asr_asr_model_valid.acc.ave/test_tr_commonvoice_lid|1639|67841|78.9|13.2|7.9|3.1|24.2|96.5|
|decode_asr_asr_model_valid.acc.ave/test_tt_commonvoice_lid|4365|159202|59.3|33.7|7.1|4.6|45.3|99.3|
|decode_asr_asr_model_valid.acc.ave/test_uk_commonvoice_lid|1671|80323|76.5|17.2|6.3|3.6|27.0|97.9|
|decode_asr_asr_model_valid.acc.ave/test_zh_CN_commonvoice_lid|8273|134476|68.2|29.7|2.1|3.7|35.5|96.0|
|decode_asr_asr_model_valid.acc.ave/test_zh_HK_commonvoice_lid|2805|32994|86.3|12.7|1.0|2.5|16.3|60.2|
|decode_asr_asr_model_valid.acc.ave/test_zh_TW_commonvoice_lid|2627|21991|81.6|16.9|1.5|5.5|24.0|60.6|
