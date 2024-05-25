# RESULTS

## Wav2Gloss: Generating Interlinear Glossed Text from Speech
- Accepted to ACL 2024
- Source code for reproducing E2E SSL-finetuned model (WavLM, XLS-R)
- Paper: https://arxiv.org/abs/2403.13169


## XLS-R ASR Single-task

- parameters for `run.sh`: `lang="full"; task="transcription"; asr_config="conf/tuning/train_xls_r_conformer.yaml"`
- date: `Tue Feb  6 09:04:03 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|6169|51.8|40.6|7.6|15.9|64.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|2342|3.3|78.9|17.8|9.5|106.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|2311|2.9|75.9|21.2|8.0|105.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|2994|5.9|87.8|6.2|45.2|139.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|2227|6.6|85.7|7.7|24.0|117.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|1682|17.8|74.7|7.4|3.9|86.0|99.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|5371|4.4|86.7|8.9|22.5|118.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|5347|26.4|67.4|6.2|8.5|82.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|3539|24.4|68.2|7.3|4.8|80.4|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|3714|7.9|77.4|14.7|11.7|103.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|3385|12.3|77.4|10.2|14.3|101.9|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|7122|3.6|57.4|39.0|1.0|97.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|7103|2.2|63.1|34.8|1.7|99.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|3368|36.4|58.9|4.6|8.8|72.4|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|3320|3.1|67.6|29.3|7.7|104.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|2533|23.5|66.4|10.1|7.3|83.8|99.3|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|4363|0.0|81.2|18.8|16.8|116.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|2640|15.3|74.0|10.7|13.0|97.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|5651|1.0|84.4|14.6|8.4|107.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|5744|3.0|76.3|20.7|3.6|100.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|6049|14.3|70.8|14.8|9.2|94.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|1263|9.0|75.9|15.1|10.5|101.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|2644|1.2|94.5|4.3|31.1|129.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|7312|22.1|72.8|5.1|34.7|112.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|4533|6.7|85.2|8.0|10.4|103.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|6299|35.1|56.8|8.2|8.3|73.2|99.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|7421|5.7|89.2|5.0|31.7|125.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|3660|4.0|89.0|7.0|27.2|123.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|4019|23.2|65.7|11.1|11.0|87.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|4111|1.4|87.5|11.1|14.3|112.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|5881|5.5|92.3|2.2|43.8|138.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|5026|2.2|94.7|3.1|35.4|133.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|4361|28.5|57.1|14.4|6.2|77.7|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|1507|15.8|67.9|16.3|12.7|96.9|99.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|2394|8.3|74.0|17.7|8.4|100.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|5495|7.6|89.3|3.2|52.6|145.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|7187|36.4|54.4|9.1|6.8|70.4|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|30942|84.6|8.9|6.5|11.4|26.8|95.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|12340|57.0|15.0|28.1|6.4|49.5|99.2|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|26226|64.1|13.4|22.5|6.9|42.8|99.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|20724|60.2|21.3|18.6|13.1|53.0|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|11672|63.8|18.2|18.1|14.3|50.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|11502|79.7|6.2|14.1|4.2|24.5|97.1|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|43522|60.0|22.8|17.2|11.9|51.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|40387|80.2|8.4|11.4|5.4|25.2|98.2|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|28303|85.1|7.0|7.9|6.8|21.7|97.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|19817|57.6|15.8|26.6|8.0|50.4|99.4|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|21910|68.4|15.3|16.3|7.9|39.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|29460|45.7|17.4|36.9|4.2|58.5|99.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|30402|44.6|29.7|25.7|5.3|60.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|21173|85.7|8.7|5.6|10.6|24.9|97.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|16979|43.8|17.6|38.6|7.0|63.2|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|15114|76.5|10.3|13.2|6.3|29.8|96.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|23416|26.9|46.4|26.7|6.8|79.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|11497|57.5|15.5|27.0|7.5|50.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|28860|45.2|36.1|18.7|10.7|65.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|26390|49.2|22.1|28.7|7.8|58.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|28924|63.7|14.6|21.8|8.1|44.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|9172|73.7|10.1|16.2|8.3|34.6|99.4|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|19659|48.2|31.5|20.3|11.7|63.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|37692|74.5|11.0|14.5|13.8|39.3|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|30222|62.7|20.6|16.7|11.5|48.8|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|29665|80.7|8.5|10.8|6.1|25.3|99.2|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|38648|66.8|18.3|14.9|21.2|54.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|19566|62.3|21.7|16.0|14.6|52.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|18615|69.1|10.3|20.6|7.8|38.7|98.3|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|26221|50.7|26.9|22.3|10.3|59.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|32402|60.2|27.0|12.8|18.8|58.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|27999|48.7|31.0|20.4|12.7|64.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|20792|76.2|6.6|17.2|5.0|28.9|98.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|9493|61.5|11.0|27.5|12.1|50.7|97.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|15419|70.1|10.2|19.7|5.6|35.5|97.1|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|38841|63.2|19.9|16.9|15.1|52.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|34860|79.4|6.3|14.2|4.5|25.1|98.9|

## XLS-R Segmentation Single-task

- parameters for `run.sh`: `lang="full"; task="underlying"; asr_config="conf/tuning/train_xls_r_conformer.yaml"`
- date: `Tue Feb  6 06:30:25 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|7507|63.0|31.3|5.7|17.7|54.7|99.1|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|3289|2.9|69.4|27.7|3.0|100.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|4542|6.5|83.7|9.8|15.6|109.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|4673|4.4|90.0|5.6|37.1|132.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|2235|3.5|96.1|0.4|86.9|183.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|3598|23.3|57.4|19.3|2.7|79.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|12177|20.8|68.2|11.0|14.0|93.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|9622|49.6|41.9|8.5|6.5|56.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|7606|51.7|44.1|4.2|15.1|63.5|99.6|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|5244|7.2|71.5|21.4|8.5|101.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|5996|15.2|73.7|11.1|17.3|102.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|6903|4.1|82.0|13.9|12.5|108.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|8164|4.0|78.9|17.2|6.5|102.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|5545|69.0|27.7|3.3|6.1|37.1|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|4479|3.8|71.0|25.3|14.1|110.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|3041|22.3|66.9|10.8|6.5|84.2|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|5511|0.2|91.1|8.6|30.7|130.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|3198|14.9|71.5|13.6|8.2|93.4|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|7969|3.0|89.4|7.6|15.4|112.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|7432|2.7|87.9|9.4|7.8|105.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|7032|17.1|67.8|15.0|8.9|91.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|2924|15.7|57.6|26.7|7.9|92.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|5007|1.6|87.5|10.9|21.5|119.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|9974|17.3|72.5|10.2|22.2|105.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|8285|18.9|71.5|9.6|14.5|95.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|9818|28.1|64.2|7.8|13.6|85.5|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|10961|5.0|89.8|5.2|21.6|116.6|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|4250|4.2|92.3|3.5|45.3|141.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|6177|23.9|52.5|23.6|4.8|80.9|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|6986|1.9|78.4|19.7|10.6|108.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|8000|6.4|91.5|2.1|36.2|129.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|5328|1.6|97.8|0.6|80.9|179.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|5343|21.4|62.3|16.2|5.4|84.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|3222|17.0|60.4|22.6|9.6|92.6|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|4197|9.9|77.6|12.5|19.2|109.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|11995|5.4|82.9|11.8|12.0|106.6|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|8929|39.1|53.5|7.4|11.6|72.5|99.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|30880|90.0|2.9|7.1|15.4|25.4|96.4|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|14528|43.8|14.7|41.5|4.1|60.2|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|31305|57.9|17.1|25.0|9.0|51.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|23429|54.8|23.3|21.9|15.6|60.8|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|11719|58.9|29.6|11.5|34.4|75.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|15427|64.8|6.6|28.5|3.5|38.7|99.1|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|57400|58.6|20.0|21.4|10.5|51.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|47799|79.6|7.3|13.2|6.1|26.5|97.3|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|36229|83.5|8.3|8.2|11.0|27.6|97.3|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|23448|46.3|16.8|36.9|5.9|59.6|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|27299|55.0|18.7|26.3|8.6|53.6|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|29372|43.6|25.9|30.5|8.8|65.2|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|32113|48.7|26.3|25.0|9.3|60.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|25102|92.0|3.2|4.8|6.0|14.1|89.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|20535|37.8|20.4|41.8|7.7|69.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|16487|72.8|9.7|17.5|5.2|32.4|99.3|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|27920|27.7|44.4|28.0|7.5|79.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|13279|53.9|15.2|30.9|5.3|51.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|32249|45.0|33.9|21.1|13.4|68.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|29892|45.5|29.9|24.7|10.2|64.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|31564|61.3|14.7|24.0|7.3|46.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|12530|53.6|11.1|35.3|5.5|51.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|24900|43.2|29.6|27.2|10.4|67.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|41699|63.1|15.5|21.4|15.9|52.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|38706|60.1|20.0|19.9|10.9|50.8|99.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|40382|68.9|14.7|16.4|8.6|39.7|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|47231|57.1|23.0|20.0|16.6|59.6|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|21799|53.3|25.0|21.7|16.1|62.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|24820|54.7|10.8|34.4|3.3|48.5|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|32323|41.7|24.8|33.5|8.2|66.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|36115|56.4|27.2|16.5|17.6|61.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|28661|42.1|44.5|13.4|23.2|81.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|23281|66.7|7.7|25.5|4.5|37.8|99.4|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|14469|50.4|13.0|36.6|7.3|57.0|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|20850|53.1|17.1|29.8|9.0|55.9|98.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|51435|52.5|21.1|26.5|11.6|59.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|37016|73.7|9.8|16.5|7.9|34.2|99.2|

## XLS-R Glossing Single-task

- parameters for `run.sh`: `lang="full"; task="gloss"; asr_config="conf/tuning/train_xls_r_conformer.yaml"`
- date: `Tue Feb  6 05:58:22 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|7507|3.1|91.9|4.9|63.7|160.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|3289|0.2|84.5|15.3|18.5|118.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|6199|0.8|90.5|8.7|30.7|130.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|4672|0.1|98.9|1.0|81.8|181.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|2235|0.2|99.0|0.8|76.2|176.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|3597|0.5|77.7|21.8|7.8|107.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|12177|1.2|88.2|10.5|35.8|134.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|9622|5.0|79.8|15.2|15.3|110.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|7606|6.7|90.6|2.7|53.9|147.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|5253|0.6|73.5|26.0|9.9|109.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|5988|1.0|80.9|18.1|8.5|107.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|6903|1.2|80.6|18.1|16.4|115.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|8171|1.5|87.8|10.7|25.4|123.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|5545|3.3|91.0|5.7|57.1|153.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|4446|0.2|86.9|12.8|30.0|129.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|3033|0.3|94.7|5.0|29.5|129.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|5507|0.4|87.0|12.6|23.0|122.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|3194|1.0|83.9|15.1|12.9|111.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|7960|2.9|87.9|9.1|15.2|112.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|7432|0.1|94.6|5.3|22.2|122.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|7032|0.6|85.4|13.9|18.1|117.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|2924|0.1|82.6|17.2|16.1|116.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|5003|0.8|93.2|5.9|33.7|132.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|9974|0.7|89.5|9.9|33.4|132.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|8285|1.9|91.9|6.2|31.5|129.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|9818|2.2|81.2|16.6|10.8|108.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|11384|0.8|96.9|2.4|46.5|145.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|4249|0.4|96.6|3.1|48.3|147.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|6177|0.6|67.0|32.5|3.9|103.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|6986|0.4|91.7|7.9|27.2|126.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|7996|0.4|98.5|1.1|56.4|156.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|5327|2.8|96.1|1.1|72.8|170.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|5340|0.2|90.8|9.0|26.8|126.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|3222|0.2|75.6|24.2|22.8|122.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|4188|0.1|95.5|4.4|69.7|169.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|11972|1.0|93.9|5.2|39.1|138.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|8929|2.5|88.0|9.4|17.1|114.6|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|49128|30.5|45.0|24.5|32.2|101.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|18953|21.7|32.7|45.6|4.0|82.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|38473|30.3|43.9|25.8|17.4|87.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|29611|30.2|58.8|11.0|43.4|113.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|17848|28.7|54.2|17.1|19.1|90.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|28164|17.5|22.1|60.3|1.3|83.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|71277|31.4|44.1|24.5|16.2|84.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|64552|36.7|38.5|24.8|16.0|79.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|53821|32.3|54.4|13.3|19.5|87.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|36299|21.2|27.3|51.5|5.8|84.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|39455|25.7|35.8|38.4|6.4|80.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|35931|26.8|43.3|29.9|10.7|83.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|41641|30.9|47.5|21.7|18.7|87.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|39676|40.0|42.7|17.2|35.2|95.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|25057|26.0|45.9|28.1|20.8|94.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|33303|20.4|30.8|48.8|3.8|83.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|31186|26.8|50.3|22.8|16.7|89.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|17485|24.2|37.0|38.7|6.4|82.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|53733|30.1|40.1|29.8|7.9|77.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|42015|26.3|50.6|23.1|10.7|84.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|39723|29.7|43.3|27.1|14.6|84.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|17075|23.9|31.9|44.2|5.9|82.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|34255|27.6|48.5|23.9|15.1|87.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|69159|26.7|42.3|31.0|11.9|85.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|58372|34.3|43.8|21.9|20.7|86.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|71751|27.6|33.2|39.2|5.6|78.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|64265|32.1|53.9|14.0|28.9|96.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|25587|25.5|58.8|15.7|26.5|101.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|31639|23.4|29.2|47.5|3.8|80.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|49442|27.1|46.0|26.8|8.0|80.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|55142|30.4|52.7|16.9|19.0|88.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|43422|26.9|50.3|22.8|12.6|85.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|34858|21.5|34.7|43.8|4.7|83.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|17541|23.9|29.2|46.9|8.6|84.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|25853|26.9|51.6|21.5|28.0|101.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|68291|30.2|51.7|18.1|21.5|91.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|50777|30.9|47.4|21.7|16.6|85.7|100.0|

## XLS-R Translation Single-task

- parameters for `run.sh`: `lang="full"; task="translation"; asr_config="conf/tuning/train_xls_r_conformer.yaml"`
- date: `Tue Feb  6 22:39:59 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|6575|5.9|79.2|14.9|22.9|117.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|2386|4.0|59.3|36.7|5.6|101.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|6375|3.6|68.3|28.1|7.4|103.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|4532|6.4|88.0|5.6|44.4|138.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|2313|7.5|85.3|7.1|27.2|119.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|3538|1.7|44.6|53.7|1.0|99.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|4658|4.3|58.4|37.3|2.6|98.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|7637|7.1|79.1|13.8|16.6|109.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|6214|1.9|81.3|16.8|7.7|105.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|5417|2.0|45.4|52.5|1.9|99.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|5528|4.6|53.7|41.7|2.1|97.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|5923|5.7|68.9|25.3|9.7|104.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|7309|6.8|70.9|22.3|9.5|102.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|4850|8.9|86.3|4.8|55.9|146.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|4256|8.0|72.4|19.5|19.4|111.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|3860|3.1|44.7|52.3|4.9|101.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|5243|6.7|72.9|20.4|14.3|107.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|3794|2.3|45.9|51.8|1.1|98.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|7572|6.6|68.5|24.9|6.6|100.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|5465|8.4|73.0|18.6|14.7|106.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|7071|1.6|55.6|42.8|2.2|100.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|2665|1.3|47.6|51.1|1.4|100.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|4364|8.1|74.8|17.0|21.2|113.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|5417|7.0|72.0|21.0|10.1|103.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|6862|0.9|78.1|20.9|11.7|110.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|2620|5.1|72.0|22.9|8.8|103.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|7945|8.2|75.8|16.0|22.2|114.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|5121|9.5|64.3|26.2|12.8|103.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|4645|5.8|55.4|38.8|3.0|97.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|5913|8.6|76.9|14.5|14.9|106.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|6924|9.6|83.6|6.8|43.6|134.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|5001|11.5|76.9|11.6|23.2|111.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|5115|1.4|56.8|41.8|1.8|100.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|37|0.0|73.0|27.0|29.7|129.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|4449|5.5|70.4|24.1|17.6|112.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|9394|8.5|79.0|12.5|19.7|111.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|7468|7.7|60.6|31.7|7.8|100.2|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|33713|40.1|38.3|21.6|27.2|87.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|10936|31.1|25.2|43.7|9.3|78.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|33333|35.6|23.6|40.8|8.4|72.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|23493|41.3|40.5|18.1|30.6|89.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|11621|41.3|37.7|21.0|21.0|79.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|17241|21.9|15.4|62.6|1.7|79.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|23747|31.8|20.7|47.5|4.8|73.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|41358|37.3|31.3|31.4|10.8|73.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|33240|40.2|37.3|22.5|16.5|76.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|26818|23.8|17.2|59.0|3.4|79.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|28786|28.3|20.8|50.9|3.7|75.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|30122|34.6|27.7|37.7|9.2|74.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|35688|37.0|28.8|34.2|9.6|72.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|24688|47.3|40.0|12.7|51.2|103.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|21991|37.5|30.8|31.7|14.7|77.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|18613|20.8|19.3|59.9|5.8|84.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|26895|36.0|30.7|33.3|11.6|75.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|18459|24.6|15.8|59.7|2.3|77.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|38454|34.6|26.1|39.3|6.0|71.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|28746|37.5|27.4|35.1|10.1|72.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|34010|31.7|21.9|46.5|6.3|74.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|13435|26.3|22.5|51.2|4.7|78.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|22710|38.1|30.4|31.4|14.5|76.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|28136|38.2|28.3|33.5|10.9|72.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|35902|38.4|37.2|24.4|21.8|83.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|13198|37.2|28.1|34.7|11.1|73.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|39032|40.5|31.4|28.2|17.4|77.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|27577|35.0|24.5|40.5|8.2|73.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|23061|28.6|19.6|51.8|3.9|75.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|30154|39.6|30.9|29.5|11.8|72.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|35395|43.5|37.1|19.5|29.1|85.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|24178|43.2|32.5|24.4|20.3|77.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|25702|30.0|25.4|44.7|6.8|76.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|191|22.5|47.6|29.8|22.0|99.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|23704|34.6|27.2|38.2|13.9|79.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|47851|40.2|33.1|26.6|14.0|73.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|38031|35.0|23.0|41.9|8.4|73.3|100.0|

## XLS-R Multi-task

- parameters for `run.sh`: `lang="full"; task="all"; asr_config="conf/tuning/train_xls_r_conformer.yaml"`
- date: `Sun Feb 11 13:59:49 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|7507|8.2|78.8|13.0|19.2|111.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|3289|0.3|67.2|32.5|1.3|101.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|6199|0.9|74.5|24.6|5.7|104.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|4672|0.2|87.2|12.5|17.3|117.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|2235|0.2|97.9|1.9|40.2|140.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|3597|0.0|52.2|47.8|0.2|100.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|12177|0.1|92.1|7.8|18.4|118.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|9622|5.6|78.2|16.3|4.4|98.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|7606|20.0|60.8|19.2|5.7|85.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|5253|0.1|71.1|28.8|4.2|104.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|5988|0.0|80.5|19.5|5.4|105.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|6903|0.4|71.7|28.0|3.0|102.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|8171|0.4|60.7|38.9|0.6|100.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|5545|15.4|68.2|16.4|5.1|89.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|4446|0.3|67.7|32.0|5.0|104.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|3033|0.5|80.7|18.8|3.2|102.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|5507|0.3|79.8|20.0|12.2|111.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|3194|1.2|74.9|24.0|3.0|101.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|7960|0.6|76.5|22.9|2.7|102.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|7432|0.2|77.7|22.1|1.9|101.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|7032|0.8|76.7|22.5|3.2|102.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|2924|0.0|65.1|34.9|2.7|102.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|5003|0.8|77.1|22.1|8.6|107.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|9974|1.2|84.9|13.8|12.2|110.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|8285|3.5|79.4|17.1|6.3|102.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|9818|0.6|79.4|20.0|2.3|101.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|11384|1.3|82.5|16.2|6.2|104.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|4249|0.4|89.8|9.8|16.6|116.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|6177|0.3|64.8|34.9|0.5|100.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|6986|0.1|71.7|28.2|3.7|103.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|7996|1.2|95.2|3.6|17.6|116.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|5327|1.4|95.6|3.0|33.4|132.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|5340|0.2|72.9|26.9|1.6|101.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|3222|0.2|67.0|32.8|5.6|105.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|4188|0.9|81.7|17.4|10.2|109.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|11972|0.2|87.7|12.2|10.4|110.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|8929|8.2|73.5|18.3|2.6|94.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|6169|21.5|72.9|5.5|36.2|114.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|2342|4.1|86.8|9.1|19.3|115.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|2311|0.0|98.2|1.8|110.9|210.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|2994|1.7|95.7|2.5|83.0|181.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|2227|3.6|95.2|1.3|63.5|159.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|1682|9.8|85.3|4.9|23.7|114.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|5371|0.4|99.6|0.1|161.0|260.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|5347|7.9|90.4|1.7|61.8|153.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|3539|2.8|96.4|0.7|103.8|201.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|3714|7.4|84.0|8.6|20.2|112.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|3385|1.9|95.5|2.6|54.8|152.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|7122|1.5|74.0|24.5|4.3|102.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|7103|1.3|72.4|26.3|3.9|102.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|3368|2.9|95.5|1.7|59.1|156.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|3320|1.4|81.6|17.0|22.2|120.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|2533|15.4|77.4|7.2|17.0|101.6|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|4363|0.0|88.6|11.4|41.0|141.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|2640|12.2|76.8|11.0|15.5|103.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|5651|0.4|95.5|4.2|33.6|133.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|5744|0.9|92.4|6.7|17.4|116.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|6049|7.7|83.1|9.2|14.3|106.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|1263|2.2|95.8|2.0|61.5|159.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|2644|0.3|98.7|1.0|84.1|183.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|7312|9.8|87.5|2.8|40.3|130.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|4533|4.6|94.5|0.9|79.9|175.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|6299|8.5|89.2|2.3|35.2|126.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|7421|1.1|96.9|2.0|47.0|146.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|3660|1.4|94.0|4.7|38.0|136.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|4019|15.7|75.8|8.5|18.2|102.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|4111|0.2|94.6|5.1|48.5|148.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|5881|2.9|96.4|0.6|65.1|162.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|5026|1.2|97.7|1.1|55.9|154.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|4361|24.3|66.2|9.6|10.6|86.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|1507|2.9|90.1|7.0|54.5|151.6|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|2394|2.3|94.0|3.7|67.8|165.5|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|5495|7.2|91.7|1.1|129.3|222.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|7187|17.1|76.7|6.2|11.5|94.3|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|6575|3.6|89.3|7.1|38.8|135.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|2386|1.9|73.7|24.4|11.8|109.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|6375|1.8|83.8|14.4|15.5|113.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|4532|3.2|87.4|9.4|34.8|131.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|2313|4.2|92.1|3.7|58.6|154.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|3538|0.9|69.5|29.6|5.4|104.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|4658|1.6|94.2|4.1|48.2|146.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|7637|2.1|92.1|5.8|23.9|121.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|6214|1.7|93.7|4.6|29.8|128.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|5417|2.0|77.1|20.9|11.6|109.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|5528|1.3|81.9|16.7|14.1|112.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|5923|2.6|78.6|18.8|16.6|113.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|7309|3.0|74.9|22.2|7.2|104.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|4850|5.6|86.6|7.9|25.7|120.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|4256|2.6|70.6|26.7|12.1|109.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|3860|2.8|73.1|24.1|6.6|103.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|5243|3.4|77.0|19.6|18.9|115.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|3794|1.6|65.5|32.9|3.3|101.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|7572|3.9|77.2|18.9|7.0|103.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|5465|4.0|86.1|10.0|25.9|122.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|7071|2.7|76.4|20.9|8.2|105.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|2665|0.7|69.5|29.8|7.5|106.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|4364|4.7|84.4|10.9|18.1|113.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|5417|6.6|89.1|4.3|41.8|135.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|6862|2.0|89.9|8.1|23.6|121.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|2620|1.9|92.1|6.0|21.8|119.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|7945|3.8|85.9|10.3|22.7|118.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|5121|4.9|77.1|17.9|16.7|111.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|4645|3.3|69.0|27.7|4.3|101.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|5913|2.8|81.6|15.6|12.6|109.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|6924|7.9|88.7|3.3|40.8|132.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|5001|8.5|86.5|5.0|52.2|143.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|5115|2.8|78.5|18.7|10.9|108.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|37|0.0|75.7|24.3|45.9|145.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|4449|2.2|83.0|14.8|20.7|118.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|9394|6.3|86.7|7.1|42.3|136.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|7468|1.9|90.8|7.2|19.1|117.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|7507|3.4|88.8|7.8|22.4|118.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|3289|1.1|76.8|22.1|5.0|104.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|4542|0.0|98.0|2.0|48.9|148.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|4673|1.4|93.2|5.5|28.9|127.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|2235|3.4|95.5|1.1|59.1|155.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|3598|1.6|73.0|25.4|2.8|101.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|12177|6.8|89.5|3.6|34.1|127.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|9622|2.1|87.2|10.7|6.3|104.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|7606|4.6|84.5|10.9|10.3|105.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|5244|2.5|78.3|19.2|10.2|107.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|5996|1.3|82.3|16.4|8.1|106.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|6903|0.6|78.3|21.1|5.5|104.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|8164|1.6|73.4|25.0|2.3|100.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|5545|4.3|90.5|5.2|10.1|105.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|4479|0.5|74.4|25.1|8.4|107.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|3041|1.4|91.2|7.4|13.2|111.7|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|5511|0.0|83.5|16.5|12.3|112.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|3198|2.0|80.4|17.7|6.4|104.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|7969|0.3|80.3|19.4|3.1|102.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|7432|0.9|81.1|18.1|3.3|102.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|7032|0.5|86.9|12.6|8.9|108.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|2924|1.8|62.0|36.2|2.3|100.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|5007|0.1|84.8|15.1|8.4|108.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|9974|1.8|91.9|6.4|22.7|120.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|8285|1.0|89.3|9.7|10.0|109.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|9818|2.5|84.4|13.1|5.9|103.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|10961|0.7|90.2|9.0|9.9|109.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|4250|0.6|95.1|4.3|26.1|125.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|6177|0.9|60.7|38.4|0.5|99.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|6986|0.1|79.2|20.7|4.2|104.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|8000|2.9|95.2|2.0|22.6|119.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|5328|1.3|97.5|1.2|40.5|139.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|5343|5.4|82.3|12.2|9.1|103.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|3222|2.1|68.7|29.2|6.9|104.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|4197|0.2|90.9|8.9|25.4|125.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|11995|4.3|86.2|9.5|17.8|113.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|8929|1.7|89.6|8.7|7.1|105.3|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|49128|31.7|27.3|40.9|13.8|82.0|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|18953|20.6|28.2|51.2|2.5|81.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|38473|27.6|40.3|32.1|8.4|80.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|29611|25.8|42.1|32.1|9.8|84.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|17848|24.6|43.7|31.7|7.3|82.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|28164|13.1|19.1|67.7|0.9|87.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|71277|32.1|35.9|32.0|7.8|75.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|64552|32.1|30.0|37.9|4.8|72.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|53821|40.8|21.5|37.7|4.4|63.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|36299|19.9|24.1|55.9|2.8|82.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|39455|23.0|31.8|45.2|3.7|80.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|35931|25.4|32.4|42.2|4.2|78.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|41641|23.1|32.2|44.6|2.9|79.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|39676|48.2|20.5|31.3|5.5|57.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|25057|21.6|30.6|47.8|5.8|84.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|33303|16.4|22.9|60.7|1.6|85.2|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|31186|25.8|35.1|39.1|6.5|80.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|17485|22.3|27.9|49.8|2.9|80.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|53733|24.4|28.6|47.0|3.1|78.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|42015|26.8|35.0|38.3|4.6|77.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|39723|28.0|34.3|37.7|5.6|77.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|17075|19.8|27.8|52.5|2.0|82.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|34255|25.2|33.1|41.7|4.6|79.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|69159|26.0|33.2|40.8|5.0|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|58372|29.0|30.5|40.5|5.1|76.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|71751|21.9|22.8|55.3|1.9|80.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|64265|29.0|39.1|31.9|8.1|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|25587|28.5|42.3|29.2|10.2|81.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|31639|23.7|26.9|49.4|2.9|79.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|49442|22.3|27.2|50.5|2.2|79.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|55142|26.5|44.6|28.9|6.9|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|43422|24.9|38.1|37.0|5.3|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|34858|23.3|24.5|52.2|2.4|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|17541|22.8|30.0|47.2|5.4|82.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|25853|25.7|37.9|36.4|8.8|83.1|99.3|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|68291|28.4|38.0|33.6|7.5|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|50777|34.3|31.0|34.8|5.3|71.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|30942|61.8|26.3|11.9|26.8|65.0|98.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|12340|55.6|16.5|28.0|7.0|51.4|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|26226|49.1|36.0|14.9|21.7|72.6|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|20724|47.5|38.8|13.6|26.6|79.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|11672|55.1|32.0|13.0|27.4|72.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|11502|68.5|9.9|21.7|6.1|37.7|99.5|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|43522|40.2|48.8|11.0|32.3|92.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|40387|50.9|34.5|14.6|20.0|69.1|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|28303|53.3|36.4|10.2|34.4|81.1|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|19817|54.5|20.5|25.0|11.9|57.4|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|21910|46.7|37.3|16.0|18.8|72.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|29460|41.6|27.2|31.2|8.1|66.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|30402|39.4|33.8|26.8|7.7|68.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|21173|39.0|54.5|6.5|49.9|110.8|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|16979|39.8|31.5|28.7|15.7|75.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|15114|64.1|17.7|18.1|9.5|45.3|98.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|23416|24.8|53.6|21.6|11.3|86.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|11497|47.8|20.9|31.3|8.1|60.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|28860|36.2|48.3|15.5|18.9|82.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|26390|40.8|42.1|17.2|19.7|78.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|28924|53.9|23.7|22.4|13.3|59.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|9172|52.4|25.8|21.8|11.2|58.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|19659|33.8|51.8|14.4|24.0|90.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|37692|58.8|27.3|13.9|25.5|66.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|30222|47.7|39.4|12.9|30.3|82.7|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|29665|60.8|29.3|9.9|25.6|64.8|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|38648|48.4|40.2|11.3|36.2|87.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|19566|49.4|35.8|14.8|21.2|71.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|18615|60.4|16.9|22.7|12.0|51.6|99.5|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|26221|37.0|44.2|18.8|17.4|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|32402|43.5|49.3|7.2|39.5|96.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|27999|41.7|43.4|14.9|20.9|79.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|20792|70.7|8.5|20.7|5.8|35.1|99.2|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|9493|44.0|30.0|25.9|27.9|83.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|15419|45.7|42.2|12.1|33.2|87.5|99.4|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|38841|48.3|43.0|8.8|36.7|88.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|34860|55.4|29.4|15.2|15.8|60.4|99.3|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|33713|41.2|41.5|17.3|33.3|92.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|10936|31.3|32.9|35.9|12.8|81.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|33333|40.0|36.7|23.4|17.5|77.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|23493|39.2|41.6|19.2|29.7|90.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|11621|40.5|46.8|12.8|37.1|96.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|17241|27.2|27.0|45.8|5.1|77.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|23747|35.9|42.9|21.2|18.7|82.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|41358|42.0|41.6|16.4|25.8|83.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|33240|41.4|43.9|14.6|29.5|88.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|26818|30.2|28.9|40.8|7.3|77.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|28786|31.5|36.6|31.8|13.1|81.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|30122|35.2|33.8|31.0|14.4|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|35688|33.8|34.1|32.1|9.6|75.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|24688|44.3|41.3|14.3|33.2|88.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|21991|33.1|30.8|36.2|11.7|78.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|18613|35.1|33.5|31.4|11.4|76.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|26895|36.2|34.9|28.8|16.6|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|18459|26.6|21.5|51.9|3.3|76.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|38454|36.9|34.7|28.4|9.7|72.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|28746|39.9|38.7|21.4|20.3|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|34010|39.1|33.2|27.6|14.5|75.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|13435|26.5|29.3|44.3|6.7|80.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|22710|40.7|37.6|21.7|20.2|79.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|28136|43.2|42.4|14.4|30.0|86.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|35902|42.4|39.7|18.0|25.2|82.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|13198|38.1|40.2|21.7|18.3|80.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|39032|42.8|38.8|18.4|26.9|84.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|27577|36.8|32.5|30.7|13.9|77.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|23061|34.2|29.0|36.8|7.6|73.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|30154|39.6|35.7|24.7|16.3|76.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|35395|46.2|42.4|11.4|34.8|88.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|24178|47.5|40.3|12.2|42.9|95.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|25702|34.2|31.9|33.9|9.2|75.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|191|22.0|32.5|45.5|29.3|107.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|23704|36.7|35.7|27.6|18.0|81.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|47851|41.1|40.1|18.8|24.7|83.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|38031|43.0|38.9|18.1|22.1|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|30880|45.9|41.4|12.7|38.2|92.3|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|14528|39.1|24.1|36.8|6.1|67.1|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|31305|36.8|44.4|18.9|17.9|81.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|23429|40.3|45.5|14.2|26.6|86.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|11719|53.7|36.2|10.2|34.0|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|15427|44.2|17.5|38.2|3.6|59.3|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|57400|46.9|32.8|20.3|14.1|67.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|47799|35.8|45.2|19.1|13.9|78.1|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|36229|39.6|44.4|16.0|18.4|78.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|23448|38.1|29.8|32.0|10.3|72.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|27299|39.7|36.2|24.1|12.3|72.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|29372|37.4|35.9|26.8|12.7|75.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|32113|40.6|35.7|23.7|10.4|69.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|25102|39.1|50.6|10.3|28.2|89.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|20535|32.8|35.5|31.7|14.7|81.9|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|16487|42.6|36.8|20.5|10.9|68.2|99.4|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|27920|27.5|46.6|26.0|11.5|84.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|13279|36.2|29.0|34.8|7.5|71.3|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|32249|36.1|45.3|18.7|16.2|80.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|29892|37.0|44.5|18.5|14.6|77.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|31564|40.2|40.4|19.3|17.1|76.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|12530|47.1|15.2|37.8|4.1|57.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|24900|28.9|49.2|21.9|11.7|82.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|41699|43.6|41.9|14.4|28.6|84.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|38706|35.4|45.9|18.7|15.7|80.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|40382|43.8|33.7|22.5|12.9|69.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|47231|40.7|46.3|13.0|23.7|83.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|21799|40.5|43.1|16.4|23.4|82.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|24820|33.0|28.5|38.5|5.1|72.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|32323|27.4|48.8|23.8|9.5|82.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|36115|41.7|49.1|9.1|31.7|90.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|28661|40.2|46.1|13.7|25.4|85.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|23281|50.3|23.3|26.4|8.2|57.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|14469|39.1|24.4|36.5|8.6|69.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|20850|33.4|48.4|18.1|22.4|89.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|51435|43.9|37.9|18.2|18.8|74.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|37016|38.5|46.9|14.7|19.3|80.8|100.0|

## WavLM ASR Single-task

- parameters for `run.sh`: `lang="full"; task="transcription"; asr_config="conf/tuning/train_wavlm_conformer.yaml"`
- date: `Tue Feb  6 12:54:24 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|6169|56.2|38.1|5.7|16.7|60.5|98.1|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|2342|3.0|77.5|19.6|9.0|106.0|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|2311|4.1|72.7|23.2|6.0|101.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|2994|6.0|90.0|4.0|54.7|148.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|2227|6.6|86.8|6.6|23.6|116.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|1682|16.3|75.1|8.6|4.8|88.5|99.2|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|5371|4.4|86.8|8.8|17.5|113.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|5347|33.5|60.5|6.1|5.1|71.6|99.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|3539|21.8|70.3|7.9|6.2|84.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|3714|7.5|77.5|15.0|11.3|103.7|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|3385|9.8|81.7|8.4|20.1|110.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|7122|3.6|57.7|38.7|0.8|97.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|7103|1.7|61.9|36.4|1.5|99.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|3368|44.4|51.2|4.4|4.0|59.6|98.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|3320|3.3|68.2|28.5|8.9|105.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|2533|25.1|63.1|11.8|6.0|80.9|98.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|4363|0.0|86.1|13.8|18.8|118.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|2640|14.3|73.2|12.5|11.9|97.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|5651|1.3|83.9|14.8|6.1|104.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|5744|3.8|76.9|19.3|3.2|99.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|6049|13.7|75.0|11.3|11.0|97.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|1263|7.4|80.5|12.1|14.6|107.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|2644|1.9|94.9|3.1|28.5|126.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|7312|16.8|76.3|6.9|26.5|109.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|4533|6.8|83.3|9.8|8.2|101.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|6299|31.3|60.7|8.0|10.3|78.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|7421|6.0|89.2|4.7|27.5|121.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|3660|3.4|90.4|6.1|26.9|123.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|4019|18.2|71.6|10.2|12.4|94.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|4111|1.9|89.6|8.5|15.7|113.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|5881|7.1|88.2|4.7|26.6|119.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|5026|2.9|92.9|4.2|24.4|121.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|4361|24.9|61.9|13.2|6.7|81.8|99.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|1507|14.2|70.7|15.1|17.8|103.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|2394|8.1|77.0|14.9|12.5|104.4|99.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|5495|9.4|88.5|2.1|58.6|149.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|7187|35.0|54.8|10.1|6.1|71.0|99.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|30942|85.1|9.0|5.9|12.0|26.9|96.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|12340|51.8|15.3|32.8|5.4|53.6|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|26226|65.8|13.3|20.9|7.3|41.5|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|20724|58.8|22.4|18.7|13.2|54.3|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|11672|64.0|17.8|18.1|14.2|50.2|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|11502|79.7|6.4|13.9|4.8|25.1|96.2|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|43522|60.1|21.9|18.0|10.2|50.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|40387|83.7|6.9|9.3|5.8|22.1|98.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|28303|83.8|8.1|8.1|8.5|24.7|99.3|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|19817|56.2|16.2|27.6|9.0|52.7|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|21910|64.3|18.4|17.3|9.5|45.2|99.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|29460|44.5|18.2|37.3|3.7|59.2|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|30402|44.3|28.8|26.9|4.8|60.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|21173|89.4|5.3|5.3|5.8|16.4|81.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|16979|43.4|17.1|39.5|7.3|63.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|15114|78.7|8.6|12.7|5.1|26.5|93.8|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|23416|27.2|48.7|24.1|7.9|80.7|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|11497|56.8|17.0|26.2|8.6|51.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|28860|45.1|35.5|19.4|9.0|63.9|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|26390|50.6|24.4|25.0|8.7|58.1|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|28924|63.9|15.9|20.2|9.4|45.5|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|9172|70.0|11.9|18.1|9.2|39.2|99.4|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|19659|50.0|30.0|20.0|11.5|61.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|37692|72.8|12.1|15.2|14.2|41.5|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|30222|62.7|22.3|15.0|11.8|49.1|99.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|29665|79.1|9.7|11.2|6.6|27.5|99.5|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|38648|66.2|19.3|14.5|20.8|54.6|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|19566|60.0|23.5|16.5|15.7|55.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|18615|65.7|12.2|22.1|9.2|43.5|98.6|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|26221|53.1|25.8|21.1|10.5|57.4|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|32402|61.2|23.1|15.6|14.0|52.8|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|27999|49.6|29.8|20.7|11.5|62.0|100.0|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|20792|73.2|8.4|18.4|5.7|32.5|98.7|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|9493|55.6|14.1|30.3|13.1|57.5|99.1|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|15419|68.0|11.9|20.1|6.4|38.4|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|38841|62.2|22.4|15.4|15.0|52.8|99.9|
|decode_transformer_lm_lm_transcription_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|34860|78.7|7.0|14.3|5.3|26.7|98.9|

## WavLM Segmentation Single-task

- parameters for `run.sh`: `lang="full"; task="underlying"; asr_config="conf/tuning/train_wavlm_conformer.yaml"`
- date: `Tue Feb  6 06:17:11 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|7507|55.0|40.6|4.4|20.0|65.0|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|3289|2.4|77.0|20.6|4.5|102.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|4542|4.5|90.9|4.6|25.2|120.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|4673|4.9|85.3|9.8|27.0|122.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|2235|4.3|95.3|0.5|79.7|175.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|3598|17.1|64.4|18.5|3.4|86.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|12177|16.6|75.3|8.1|15.3|98.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|9622|49.1|43.6|7.4|8.4|59.4|99.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|7606|50.8|44.8|4.4|13.0|62.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|5244|5.3|76.5|18.2|11.4|106.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|5996|12.4|78.9|8.8|20.1|107.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|6903|3.2|84.2|12.6|11.3|108.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|8164|4.5|78.3|17.2|5.8|101.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|5545|66.8|29.9|3.3|6.1|39.3|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|4479|3.2|70.0|26.7|10.7|107.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|3041|17.7|76.6|5.8|15.0|97.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|5511|0.0|91.6|8.4|31.5|131.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|3198|12.7|76.5|10.8|10.2|97.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|7969|2.9|84.0|13.1|10.1|107.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|7432|3.4|87.1|9.5|8.5|105.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|7032|11.5|77.3|11.2|11.0|99.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|2924|12.0|63.6|24.3|8.5|96.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|5007|2.0|85.3|12.7|19.1|117.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|9974|12.3|76.7|11.1|18.1|105.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|8285|15.1|77.7|7.2|18.1|103.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|9818|21.7|71.7|6.7|15.4|93.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|10961|4.4|85.3|10.3|12.5|108.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|4250|4.1|91.6|4.2|42.4|138.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|6177|13.7|69.8|16.5|7.5|93.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|6986|1.7|76.7|21.6|7.7|106.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|8000|5.5|88.3|6.2|20.6|115.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|5328|1.6|96.8|1.6|66.0|164.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|5343|16.8|67.7|15.5|6.6|89.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|3222|10.9|71.0|18.1|11.6|100.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|4197|8.1|79.9|11.9|25.9|117.8|99.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|11995|6.2|82.3|11.5|12.1|105.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|8929|29.8|63.7|6.5|13.7|83.8|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|30880|88.0|4.5|7.4|15.2|27.2|95.3|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|14528|43.5|16.8|39.6|4.2|60.6|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|31305|55.8|18.2|26.1|7.5|51.8|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|23429|50.1|20.1|29.8|10.1|60.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|11719|58.7|28.8|12.5|27.9|69.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|15427|64.0|7.4|28.6|3.0|39.0|99.4|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|57400|56.3|20.1|23.6|8.5|52.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|47799|79.0|7.9|13.1|6.4|27.4|98.2|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|36229|83.8|8.2|8.0|10.0|26.2|98.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|23448|44.6|18.7|36.7|6.0|61.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|27299|53.9|19.2|26.9|8.3|54.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|29372|44.5|23.6|31.9|7.6|63.1|99.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|32113|49.0|25.1|25.9|7.5|58.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|25102|92.4|3.6|4.0|6.1|13.7|79.7|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|20535|36.7|18.2|45.0|5.9|69.2|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|16487|70.7|11.6|17.6|6.4|35.7|95.5|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|27920|27.2|43.0|29.7|6.3|79.1|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|13279|52.6|16.3|31.1|5.2|52.6|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|32249|43.2|32.5|24.3|10.2|67.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|29892|46.1|28.4|25.5|8.6|62.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|31564|56.6|17.5|25.9|7.1|50.5|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|12530|51.0|12.7|36.2|6.3|55.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|24900|43.1|26.2|30.8|8.3|65.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|41699|60.3|15.7|24.0|13.9|53.7|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|38706|58.7|22.0|19.3|12.0|53.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|40382|65.4|16.9|17.7|8.6|43.2|99.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|47231|54.4|20.1|25.5|11.3|56.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|21799|53.0|24.1|23.0|14.3|61.4|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|24820|50.3|15.2|34.6|3.6|53.3|99.8|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|32323|41.4|22.8|35.9|6.7|65.3|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|36115|53.7|22.8|23.6|11.7|58.0|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|28661|42.3|40.3|17.4|17.5|75.2|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|23281|64.5|9.2|26.3|4.8|40.4|99.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|14469|46.5|15.5|38.1|7.1|60.6|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|20850|52.5|19.4|28.2|11.7|59.3|99.9|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|51435|52.5|19.1|28.4|9.4|56.9|100.0|
|decode_transformer_lm_lm_underlying_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|37016|69.7|12.9|17.5|7.9|38.2|99.3|

## WavLM Glossing Single-task

- parameters for `run.sh`: `lang="full"; task="gloss"; asr_config="conf/tuning/train_wavlm_conformer.yaml"`
- date: `Tue Feb  6 10:46:55 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|7507|3.1|93.2|3.6|125.8|222.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|3289|0.5|71.1|28.4|4.3|103.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|6199|0.4|94.0|5.5|58.4|157.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|4672|0.1|97.9|2.0|105.8|205.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|2235|0.1|99.4|0.4|100.6|200.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|3597|0.9|60.0|39.0|1.0|100.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|12177|0.7|92.0|7.2|85.7|184.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|9622|2.9|89.1|8.0|41.6|138.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|7606|4.8|92.2|3.0|73.2|168.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|5253|0.2|69.1|30.7|11.2|111.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|5988|0.9|84.2|15.0|20.3|119.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|6903|0.8|83.9|15.3|26.1|125.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|8171|1.0|88.9|10.1|26.0|125.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|5545|5.9|85.5|8.6|32.6|126.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|4446|0.6|89.2|10.2|40.0|139.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|3033|0.7|94.4|5.0|37.0|136.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|5507|0.7|90.4|9.0|38.7|138.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|3194|1.5|79.9|18.6|11.8|110.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|7960|1.9|91.2|6.9|29.2|127.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|7432|0.2|93.5|6.3|46.1|145.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|7032|0.4|87.6|11.9|26.7|126.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|2924|0.1|69.8|30.1|12.2|112.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|5003|2.2|90.9|6.9|41.6|139.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|9974|0.7|85.2|14.1|43.8|143.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|8285|2.0|92.7|5.3|44.7|142.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|9818|1.0|88.8|10.2|25.6|124.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|11384|0.6|96.2|3.2|66.6|166.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|4249|0.7|95.4|3.9|51.1|150.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|6177|0.1|65.9|34.1|2.8|102.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|6986|0.8|92.8|6.4|55.0|154.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|7996|1.4|91.6|7.0|68.0|166.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|5327|1.6|96.8|1.5|114.2|212.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|5340|0.3|77.0|22.6|11.7|111.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|3222|0.1|66.2|33.7|14.2|114.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|4188|0.1|92.2|7.7|44.5|144.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|11972|0.7|90.9|8.4|56.2|155.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|8929|0.9|88.4|10.7|22.5|121.6|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|49128|27.0|49.1|23.9|28.7|101.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|18953|22.8|30.8|46.4|4.6|81.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|38473|30.3|48.2|21.5|18.6|88.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|29611|30.6|57.8|11.6|42.5|111.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|17848|25.2|56.6|18.1|16.2|91.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|28164|18.4|19.8|61.8|1.8|83.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|71277|29.5|45.6|24.9|13.1|83.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|64552|31.7|38.0|30.4|9.7|78.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|53821|33.7|45.8|20.5|15.6|81.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|36299|21.0|27.6|51.3|5.7|84.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|39455|26.4|32.9|40.7|5.4|79.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|35931|26.6|43.9|29.5|10.4|83.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|41641|30.2|45.2|24.6|13.9|83.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|39676|47.8|34.6|17.5|32.0|84.2|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|25057|27.1|46.2|26.8|23.4|96.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|33303|20.4|29.5|50.1|3.2|82.8|99.1|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|31186|27.2|48.4|24.4|14.2|87.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|17485|25.0|35.7|39.2|6.3|81.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|53733|28.5|36.6|35.0|6.0|77.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|42015|23.8|48.9|27.3|5.7|81.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|39723|29.3|44.9|25.8|15.6|86.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|17075|22.5|34.2|43.2|5.5|83.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|34255|30.1|40.0|29.9|10.9|80.8|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|69159|26.5|42.1|31.3|9.1|82.6|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|58372|35.6|40.4|24.0|23.3|87.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|71751|25.5|30.5|43.9|4.5|78.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|64265|28.3|54.4|17.3|19.6|91.3|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|25587|28.6|55.3|16.1|28.6|100.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|31639|23.4|29.8|46.7|3.9|80.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|49442|27.0|42.1|30.9|6.0|79.0|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|55142|27.6|48.0|24.4|13.1|85.5|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|43422|23.7|50.9|25.4|9.1|85.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|34858|23.5|36.0|40.5|6.3|82.9|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|17541|21.3|33.9|44.7|12.0|90.7|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|25853|29.2|47.4|23.4|28.3|99.1|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|68291|29.0|53.5|17.5|18.4|89.4|100.0|
|decode_transformer_lm_lm_gloss_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|50777|28.5|45.1|26.4|11.7|83.2|100.0|

## WavLM Translation Single-task

- parameters for `run.sh`: `lang="full"; task="translation"; asr_config="conf/tuning/train_wavlm_conformer.yaml"`
- date: `Tue Feb  6 09:25:28 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|6575|6.0|57.1|36.9|4.9|98.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|2386|2.3|40.9|56.8|1.7|99.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|6375|4.0|47.3|48.7|1.5|97.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|4532|5.0|58.0|37.0|4.9|99.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|2313|4.9|51.4|43.8|2.5|97.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|3538|0.3|34.0|65.7|0.1|99.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|4658|5.2|42.1|52.8|0.9|95.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|7637|5.5|49.3|45.2|1.9|96.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|6214|6.8|68.3|24.9|5.3|98.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|5417|3.7|32.8|63.6|1.3|97.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|5528|3.2|49.0|47.7|1.2|98.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|5923|3.7|41.8|54.6|1.9|98.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|7309|4.4|39.2|56.4|0.8|96.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|4850|5.1|70.3|24.7|8.2|103.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|4256|4.6|45.9|49.4|3.5|98.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|3860|2.9|35.9|61.3|0.8|97.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|5243|3.6|45.2|51.2|1.8|98.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|3794|5.6|42.9|51.5|2.5|96.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|7572|4.9|36.1|59.0|0.4|95.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|5465|7.2|39.8|53.0|1.6|94.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|7071|5.0|41.9|53.1|1.1|96.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|2665|3.0|43.2|53.8|1.2|98.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|4364|6.2|43.1|50.7|1.8|95.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|5417|5.8|38.6|55.6|1.2|95.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|6862|7.7|61.7|30.6|5.8|98.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|2620|4.3|48.6|47.1|1.5|97.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|7945|6.6|43.8|49.6|2.3|95.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|5121|4.7|38.1|57.2|1.2|96.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|4645|3.9|42.5|53.6|1.2|97.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|5913|5.5|43.1|51.4|1.3|95.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|6924|8.3|52.9|38.8|4.1|95.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|5001|7.8|46.8|45.4|2.8|95.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|5115|5.5|37.0|57.6|0.8|95.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|37|0.0|62.2|37.8|5.4|105.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|4449|5.1|50.7|44.2|6.3|101.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|9394|6.5|44.9|48.7|1.8|95.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|7468|5.7|46.6|47.7|1.6|96.0|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|33713|30.9|22.2|46.9|5.4|74.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|10936|19.4|16.7|63.9|3.8|84.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|33333|28.8|15.2|56.0|2.5|73.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|23493|31.4|23.6|45.0|6.7|75.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|11621|29.7|21.6|48.7|4.1|74.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|17241|18.1|13.2|68.7|1.0|82.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|23747|28.1|15.2|56.7|2.2|74.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|41358|28.2|15.4|56.4|2.5|74.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|33240|35.4|23.3|41.3|5.3|69.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|26818|19.1|11.5|69.4|1.5|82.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|28786|24.3|16.2|59.5|2.0|77.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|30122|24.5|16.7|58.8|2.7|78.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|35688|26.1|14.8|59.1|2.0|75.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|24688|36.5|27.1|36.4|10.2|73.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|21991|26.9|19.0|54.1|4.0|77.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|18613|24.4|16.1|59.5|3.1|78.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|26895|25.7|18.1|56.2|2.9|77.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|18459|24.6|14.5|60.9|2.3|77.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|38454|24.9|12.1|63.1|1.1|76.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|28746|27.7|13.1|59.1|2.3|74.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|34010|26.8|14.1|59.1|2.3|75.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|13435|22.8|14.4|62.9|1.7|78.9|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|22710|28.3|14.9|56.8|2.8|74.5|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|28136|28.1|15.1|56.9|2.2|74.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|35902|35.1|24.2|40.7|6.2|71.1|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|13198|28.0|15.1|56.8|2.6|74.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|39032|30.0|16.6|53.4|3.6|73.6|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|27577|25.0|11.9|63.1|1.9|77.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|23061|24.3|15.9|59.8|2.7|78.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|30154|28.3|15.4|56.3|2.3|74.0|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|35395|34.0|20.1|45.9|5.7|71.7|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|24178|31.8|18.8|49.4|5.1|73.3|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|25702|24.1|13.3|62.6|1.5|77.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|191|20.4|28.8|50.8|5.2|84.8|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|23704|29.0|18.3|52.8|6.4|77.4|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|47851|29.6|16.5|53.8|2.8|73.2|100.0|
|decode_transformer_lm_lm_translation_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|38031|28.9|14.5|56.6|3.1|74.2|100.0|

## WavLM Multi-task

- parameters for `run.sh`: `lang="full"; task="all"; asr_config="conf/tuning/train_wavlm_conformer.yaml"`
- date: `Sun Feb 11 05:03:28 EST 2024`
- python version: `3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:40:32) [GCC 12.3.0]`
- espnet version: `espnet 202310`
- pytorch version: `pytorch 2.1.0`
- Git hash: `5432ad0c13fc1a797324134223e1d9f159148722`
  - Commit date: `Mon Feb 5 16:11:26 2024 -0500`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|7507|0.7|88.5|10.8|19.2|118.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|3289|0.1|70.3|29.6|2.0|101.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|6199|1.0|79.2|19.8|9.8|108.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|4672|0.2|93.8|6.0|26.0|125.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|2235|0.1|98.9|1.0|53.0|152.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|3597|0.0|50.5|49.5|0.1|100.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|12177|0.0|93.8|6.2|18.4|118.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|9622|4.9|77.7|17.4|4.2|99.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|7606|9.3|77.5|13.2|11.8|102.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|5253|0.1|72.4|27.5|5.0|104.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|5988|0.1|86.2|13.7|10.2|110.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|6903|0.0|81.4|18.6|5.8|105.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|8171|0.2|73.8|25.9|1.4|101.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|5545|21.2|66.5|12.3|4.9|83.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|4446|0.1|72.8|27.1|8.0|107.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|3033|0.1|83.7|16.2|4.2|104.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|5507|0.2|90.5|9.4|23.1|122.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|3194|0.2|86.8|13.0|12.8|112.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|7960|0.1|90.7|9.2|10.8|110.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|7432|0.1|89.2|10.7|4.9|104.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|7032|0.1|88.3|11.7|10.6|110.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|2924|0.0|64.5|35.5|2.8|102.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|5003|0.4|87.6|12.0|12.1|111.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|9974|0.3|93.1|6.6|18.7|118.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|8285|0.3|92.3|7.4|17.6|117.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|9818|0.3|79.0|20.7|2.2|101.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|11384|0.3|88.5|11.1|7.8|107.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|4249|0.1|94.0|5.9|24.5|124.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|6177|0.1|76.8|23.2|2.2|102.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|6986|0.1|84.7|15.1|8.8|108.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|7996|0.2|97.4|2.4|28.0|127.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|5327|0.4|98.1|1.5|44.1|143.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|5340|0.1|83.0|16.9|4.2|104.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|3222|0.7|64.3|35.0|4.7|104.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|4188|0.1|81.9|18.0|11.0|110.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|11972|0.1|89.8|10.1|11.7|111.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|8929|0.3|88.1|11.6|5.7|105.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|6169|25.0|72.3|2.6|50.9|125.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|2342|2.7|91.4|5.8|29.3|126.6|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|2311|0.0|99.4|0.6|178.2|278.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|2994|5.2|94.1|0.7|122.4|217.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|2227|3.2|96.6|0.2|65.8|162.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|1682|0.8|95.9|3.3|62.6|161.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|5371|0.1|99.8|0.1|167.8|267.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|5347|0.5|98.5|1.0|71.6|171.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|3539|0.5|99.4|0.1|127.2|226.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|3714|2.3|93.8|3.9|36.4|134.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|3385|0.6|97.8|1.5|77.5|176.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|7122|1.3|85.4|13.3|6.1|104.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|7103|2.2|90.4|7.4|15.3|113.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|3368|1.9|97.5|0.6|64.4|162.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|3320|1.5|87.7|10.8|36.1|134.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|2533|3.3|94.8|1.9|42.5|139.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|4363|0.0|96.9|3.1|50.3|150.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|2640|2.3|88.6|9.1|18.7|116.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|5651|0.5|98.7|0.8|40.8|140.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|5744|1.0|96.7|2.3|27.1|126.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|6049|1.6|94.8|3.6|26.7|125.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|1263|0.7|98.9|0.4|122.2|221.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|2644|0.3|99.6|0.1|99.4|199.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|7312|2.7|96.8|0.4|74.8|172.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|4533|1.1|98.5|0.4|99.3|198.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|6299|5.2|93.9|0.9|43.1|138.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|7421|0.8|98.9|0.2|60.5|159.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|3660|2.0|96.9|1.0|61.1|159.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|4019|4.1|91.3|4.6|29.0|125.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|4111|0.1|99.2|0.7|71.6|171.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|5881|2.4|97.4|0.2|76.2|173.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|5026|1.1|98.8|0.1|65.3|164.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|4361|5.7|90.9|3.4|26.1|120.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|1507|3.5|95.0|1.5|100.3|196.9|99.5|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|2394|0.3|97.4|2.3|121.1|220.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|5495|8.0|91.2|0.9|127.8|219.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|7187|3.5|95.6|0.9|35.4|131.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|6575|3.1|88.8|8.1|39.6|136.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|2386|0.8|74.1|25.2|13.3|112.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|6375|3.4|82.5|14.1|16.2|112.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|4532|2.4|90.5|7.0|45.4|143.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|2313|3.8|93.6|2.7|61.3|157.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|3538|0.3|56.2|43.6|1.0|100.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|4658|1.0|93.9|5.2|35.9|135.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|7637|3.1|89.0|7.9|22.0|118.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|6214|1.4|92.3|6.3|34.7|133.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|5417|0.9|73.8|25.3|8.9|108.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|5528|2.1|86.0|11.9|22.9|120.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|5923|2.5|82.3|15.2|20.9|118.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|7309|2.1|84.7|13.1|14.0|111.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|4850|3.7|85.3|11.1|17.6|113.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|4256|1.1|76.7|22.2|14.9|113.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|3860|1.0|70.2|28.8|4.5|103.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|5243|2.9|83.8|13.3|28.3|125.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|3794|0.8|72.0|27.3|5.7|104.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|7572|3.4|87.2|9.4|14.4|111.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|5465|2.9|92.1|5.0|33.7|130.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|7071|2.4|80.7|16.9|11.2|108.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|2665|0.2|65.2|34.6|7.7|107.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|4364|3.8|88.7|7.5|24.9|121.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|5417|3.4|92.6|3.9|43.4|140.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|6862|3.2|91.4|5.4|38.6|135.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|2620|1.8|91.4|6.9|20.7|119.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|7945|3.5|88.9|7.6|29.4|125.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|5121|1.7|84.6|13.7|24.1|122.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|4645|2.0|84.9|13.1|16.9|114.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|5913|3.0|88.6|8.4|25.0|122.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|6924|4.1|94.2|1.7|52.3|148.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|5001|6.3|90.8|2.9|65.6|159.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|5115|1.3|84.6|14.1|17.2|115.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|37|0.0|75.7|24.3|35.1|135.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|4449|0.3|85.2|14.5|22.0|121.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|9394|4.9|88.4|6.7|38.6|133.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|7468|1.3|93.5|5.2|29.1|127.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|7507|21.0|69.1|9.9|26.0|105.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|3289|2.0|70.8|27.2|2.6|100.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|4542|0.2|96.1|3.7|38.8|138.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|4673|2.3|94.1|3.6|36.3|134.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|2235|2.9|96.2|0.9|59.6|156.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|3598|3.7|48.8|47.5|0.3|96.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|12177|9.9|82.4|7.8|21.9|112.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|9622|8.4|78.3|13.3|5.8|97.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|7606|3.6|84.4|12.0|12.4|108.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|5244|1.2|76.1|22.7|8.2|107.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|5996|0.8|83.3|15.9|7.6|106.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|6903|0.8|83.7|15.4|7.0|106.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|8164|1.6|84.9|13.5|6.5|104.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|5545|8.7|81.7|9.7|5.9|97.2|99.6|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|4479|1.0|75.5|23.5|11.1|110.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|3041|6.2|82.8|11.0|11.0|104.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|5511|0.0|89.3|10.7|19.5|119.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|3198|5.4|78.0|16.6|6.8|101.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|7969|0.3|88.1|11.6|6.4|106.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|7432|0.4|89.0|10.5|4.5|104.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|7032|3.4|84.6|12.0|10.8|107.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|2924|4.9|60.7|34.4|5.3|100.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|5007|0.1|88.6|11.3|12.1|112.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|9974|4.5|88.3|7.1|22.0|117.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|8285|1.2|90.6|8.2|14.7|113.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|9818|5.4|77.0|17.7|5.1|99.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|10961|0.6|93.0|6.4|13.0|112.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|4250|1.5|95.2|3.3|35.1|133.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|6177|7.0|66.2|26.7|3.1|96.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|6986|0.4|87.7|11.9|7.9|107.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|8000|1.3|97.1|1.7|26.3|125.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|5328|0.9|98.6|0.5|47.3|146.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|5343|8.3|76.3|15.4|9.2|100.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|3222|2.1|68.5|29.4|9.0|106.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|4197|2.5|84.2|13.2|19.0|116.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|11995|3.4|86.8|9.8|11.3|107.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|8929|3.6|89.9|6.5|12.3|108.7|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ainu1240_test|750|49128|26.8|28.8|44.4|11.0|84.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_apah1238_test|662|18953|20.9|27.5|51.7|2.5|81.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arap1274_test|735|38473|25.9|35.5|38.7|5.6|79.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_arta1239_test|593|29611|26.4|39.4|34.2|7.9|81.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_balk1252_test|233|17848|24.7|40.2|35.1|5.9|81.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_beja1238_test|660|28164|13.2|18.8|68.0|0.9|87.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_bora1263_test|746|71277|32.2|35.0|32.8|7.4|75.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_dolg1241_test|736|64552|31.2|30.5|38.3|4.6|73.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_even1259_test|749|53821|34.2|27.6|38.2|4.8|70.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_goro1270_test|655|36299|20.0|22.0|58.0|2.6|82.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_jeju1234_test|646|39455|23.8|27.0|49.2|2.9|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kach1280_test|748|35931|26.3|30.7|43.0|3.8|77.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kaka1265_test|712|41641|25.2|31.5|43.3|3.0|77.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kama1378_test|750|39676|43.3|21.2|35.5|4.1|60.8|99.5|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_kara1499_test|640|25057|22.6|27.5|49.9|5.0|82.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_komn1238_test|681|33303|16.1|23.0|61.0|1.5|85.4|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_mand1415_test|676|31186|27.3|35.4|37.3|6.1|78.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nngg1234_test|616|17485|22.4|28.9|48.7|3.0|80.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_nort2641_test|407|53733|24.3|27.9|47.9|2.2|78.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_pnar1238_test|252|42015|28.1|30.0|41.9|3.0|74.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_port1286_test|738|39723|27.5|33.8|38.7|4.3|76.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_ruul1235_test|397|17075|21.1|27.8|51.2|2.3|81.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sanz1248_test|305|34255|25.5|30.3|44.2|3.6|78.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_savo1255_test|709|69159|25.1|28.2|46.7|3.3|78.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_selk1253_test|708|58372|26.1|30.5|43.4|3.8|77.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_slav1254_test|743|71751|21.0|20.6|58.4|1.3|80.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sout2856_test|589|64265|27.9|34.0|38.0|4.7|76.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumb1241_test|461|25587|27.7|41.1|31.2|8.0|80.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_sumi1235_test|655|31639|24.3|26.6|49.1|2.4|78.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taba1259_test|498|49442|23.3|26.3|50.4|1.9|78.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_taul1251_test|380|55142|26.1|36.3|37.5|4.1|78.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tehr1242_test|409|43422|24.4|35.2|40.4|4.0|79.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_teop1238_test|722|34858|24.0|26.0|50.0|2.6|78.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_texi1237_test|569|17541|21.1|26.1|52.8|3.4|82.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_tond1251_test|714|25853|23.3|33.5|43.3|5.1|81.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_trin1278_test|694|68291|27.6|33.3|39.1|4.7|77.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_gloss_vera1241_test|727|50777|27.2|34.1|38.7|4.0|76.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ainu1240_test|750|30942|64.5|23.6|11.9|22.0|57.5|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_apah1238_test|649|12340|40.8|26.5|32.7|5.9|65.1|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arap1274_test|729|26226|34.7|46.8|18.6|16.6|81.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_arta1239_test|593|20724|47.3|40.2|12.5|26.9|79.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_balk1252_test|233|11672|53.1|33.5|13.4|26.2|73.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_beja1238_test|658|11502|49.0|26.0|25.0|8.7|59.8|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_bora1263_test|744|43522|34.2|56.2|9.6|33.2|99.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_dolg1241_test|736|40387|36.7|49.9|13.4|22.3|85.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_even1259_test|749|28303|41.1|50.5|8.4|36.0|94.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_goro1270_test|653|19817|39.5|29.9|30.6|10.7|71.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_jeju1234_test|643|21910|34.3|46.1|19.6|13.8|79.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kach1280_test|748|29460|43.6|23.9|32.5|8.6|65.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kaka1265_test|711|30402|40.6|36.5|22.9|11.4|70.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kama1378_test|750|21173|36.9|55.9|7.2|45.7|108.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_kara1499_test|640|16979|38.1|32.1|29.8|13.1|75.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_komn1238_test|682|15114|51.3|29.5|19.2|11.8|60.5|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_mand1415_test|672|23416|28.0|54.3|17.7|13.6|85.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nngg1234_test|600|11497|38.4|29.7|31.9|10.7|72.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_nort2641_test|407|28860|40.7|45.7|13.6|21.7|81.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_pnar1238_test|252|26390|43.7|38.3|18.0|18.1|74.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_port1286_test|737|28924|41.7|38.4|20.0|14.3|72.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_ruul1235_test|348|9172|41.2|39.3|19.4|16.0|74.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sanz1248_test|305|19659|35.0|50.4|14.6|20.1|85.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_savo1255_test|709|37692|46.4|39.3|14.3|24.0|77.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_selk1253_test|708|30222|41.8|47.2|10.9|32.9|91.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_slav1254_test|743|29665|53.6|35.4|11.0|25.7|72.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sout2856_test|589|38648|47.7|40.2|12.1|30.4|82.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumb1241_test|461|19566|50.3|32.8|16.9|20.8|70.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_sumi1235_test|651|18615|42.4|33.1|24.5|12.4|70.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taba1259_test|498|26221|37.8|46.4|15.7|18.3|80.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_taul1251_test|380|32402|45.3|44.4|10.3|29.4|84.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tehr1242_test|409|27999|43.0|42.5|14.5|19.5|76.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_teop1238_test|719|20792|55.0|20.9|24.2|10.6|55.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_texi1237_test|549|9493|39.2|34.7|26.1|24.2|85.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_tond1251_test|714|15419|39.9|48.4|11.7|26.5|86.6|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_trin1278_test|694|38841|51.0|39.8|9.2|34.6|83.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_transcription_vera1241_test|727|34860|44.0|37.5|18.5|14.0|70.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ainu1240_test|748|33713|35.7|41.0|23.3|24.2|88.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_apah1238_test|514|10936|25.9|33.8|40.4|8.7|82.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arap1274_test|719|33333|36.3|33.5|30.2|11.6|75.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_arta1239_test|583|23493|35.6|42.6|21.8|23.9|88.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_balk1252_test|233|11621|36.4|48.1|15.5|29.3|92.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_beja1238_test|651|17241|21.0|28.0|51.0|3.2|82.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_bora1263_test|367|23747|32.9|44.2|22.9|18.1|85.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_dolg1241_test|729|41358|38.5|42.8|18.8|23.6|85.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_even1259_test|741|33240|36.0|46.4|17.7|25.3|89.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_goro1270_test|640|26818|26.7|27.8|45.5|5.0|78.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_jeju1234_test|607|28786|30.1|33.5|36.4|8.5|78.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kach1280_test|748|30122|30.6|34.7|34.6|9.8|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kaka1265_test|710|35688|32.1|35.8|32.1|8.3|76.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kama1378_test|735|24688|38.7|45.0|16.3|32.5|93.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_kara1499_test|620|21991|27.9|31.1|41.0|7.9|80.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_komn1238_test|672|18613|28.1|35.8|36.1|8.3|80.2|99.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_mand1415_test|667|26895|33.2|36.7|30.1|14.8|81.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nngg1234_test|594|18459|23.3|23.0|53.7|2.3|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_nort2641_test|407|38454|35.7|35.8|28.5|9.5|73.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_pnar1238_test|252|28746|37.9|39.4|22.7|17.1|79.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_port1286_test|697|34010|34.0|32.5|33.5|9.2|75.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_ruul1235_test|349|13435|26.1|28.9|44.9|6.8|80.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sanz1248_test|303|22710|37.7|38.1|24.2|16.6|79.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_savo1255_test|443|28136|37.0|40.8|22.2|16.8|79.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_selk1253_test|694|35902|37.0|41.2|21.9|18.9|82.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_slav1254_test|267|13198|33.6|40.7|25.7|13.5|79.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sout2856_test|468|39032|38.6|38.9|22.5|19.5|80.9|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumb1241_test|455|27577|30.2|34.7|35.1|9.2|79.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_sumi1235_test|638|23061|30.6|31.1|38.3|6.8|76.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taba1259_test|497|30154|37.0|38.4|24.5|15.5|78.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_taul1251_test|377|35395|38.3|46.6|15.0|23.3|85.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tehr1242_test|407|24178|41.3|44.6|14.1|35.4|94.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_teop1238_test|713|25702|29.8|31.7|38.5|6.5|76.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_texi1237_test|14|191|21.5|34.6|44.0|20.4|99.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_tond1251_test|654|23704|26.1|36.5|37.4|8.3|82.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_trin1278_test|690|47851|39.8|39.8|20.4|22.5|82.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_translation_vera1241_test|714|38031|33.4|42.0|24.6|12.5|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ainu1240_test|750|30880|62.8|22.1|15.1|27.1|64.3|98.9|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_apah1238_test|662|14528|40.5|19.6|40.0|3.8|63.3|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arap1274_test|735|31305|37.9|37.4|24.7|12.2|74.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_arta1239_test|593|23429|42.1|41.1|16.8|20.3|78.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_balk1252_test|233|11719|49.9|37.7|12.5|26.1|76.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_beja1238_test|660|15427|49.4|9.8|40.8|1.9|52.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_bora1263_test|746|57400|48.0|31.1|20.9|14.1|66.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_dolg1241_test|736|47799|44.2|35.6|20.2|12.0|67.8|99.7|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_even1259_test|749|36229|43.3|39.9|16.8|18.0|74.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_goro1270_test|655|23448|38.0|25.4|36.6|7.2|69.3|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_jeju1234_test|648|27299|35.0|36.1|28.9|8.6|73.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kach1280_test|748|29372|41.4|27.4|31.2|9.2|67.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kaka1265_test|712|32113|42.0|33.0|25.0|9.7|67.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kama1378_test|750|25102|46.2|42.1|11.7|25.4|79.3|98.4|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_kara1499_test|640|20535|33.7|29.3|37.0|9.8|76.1|99.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_komn1238_test|684|16487|53.0|23.8|23.3|7.7|54.7|98.8|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_mand1415_test|676|27920|28.0|46.7|25.3|9.0|81.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nngg1234_test|620|13279|41.0|24.1|34.9|6.0|65.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_nort2641_test|407|32249|38.4|43.8|17.8|16.1|77.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_pnar1238_test|252|29892|41.0|36.9|22.1|11.6|70.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_port1286_test|739|31564|46.2|28.9|25.0|10.3|64.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_ruul1235_test|397|12530|47.5|16.6|35.9|5.9|58.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sanz1248_test|305|24900|31.0|43.6|25.4|9.2|78.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_savo1255_test|709|41699|47.9|31.5|20.6|20.6|72.7|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_selk1253_test|708|38706|36.0|42.9|21.2|13.6|77.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_slav1254_test|743|40382|47.7|24.6|27.7|8.3|60.6|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sout2856_test|590|47231|42.7|38.6|18.7|15.7|73.0|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumb1241_test|461|21799|44.7|33.4|21.8|16.2|71.4|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_sumi1235_test|655|24820|41.3|20.4|38.2|3.6|62.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taba1259_test|498|32323|32.1|44.1|23.8|9.1|77.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_taul1251_test|380|36115|42.2|42.6|15.2|21.3|79.1|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tehr1242_test|409|28661|40.8|44.6|14.6|20.0|79.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_teop1238_test|722|23281|54.5|15.9|29.6|6.1|51.5|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_texi1237_test|569|14469|35.0|23.9|41.0|7.3|72.2|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_tond1251_test|714|20850|44.3|27.0|28.8|12.0|67.7|98.6|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_trin1278_test|694|51435|44.3|35.1|20.6|16.1|71.8|100.0|
|decode_transformer_lm_lm_all_full_4layer_valid.loss.ave_asr_model_valid.acc.ave/w2g_underlying_vera1241_test|727|37016|43.6|35.3|21.0|10.6|67.0|100.0|
