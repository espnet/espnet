# RESULTS

This is Swahili ASR system from our IWSLT 2021 Low-Resource Speech Translation submission
([paper](https://aclanthology.org/2021.iwslt-1.21.pdf)).

`_raw` results show the same system scored using unprocessed reference transcriptions
(without written to spoken language conversion).

## asr_train_asr_conformer_raw_sw_bpe100_sp
- model link: https://zenodo.org/record/5226979

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa|868|18332|92.8|6.6|0.7|5.4|12.6|68.2|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa_raw|868|18412|82.0|17.3|0.7|5.0|22.9|94.6|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc|868|19504|84.5|12.7|2.8|2.5|18.0|84.2|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc_raw|868|19512|76.2|21.4|2.5|2.1|26.0|96.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa|868|117682|98.2|1.1|0.7|4.6|6.4|68.3|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa_raw|868|119860|95.7|2.0|2.3|4.2|8.5|94.7|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc|868|119172|96.1|1.7|2.3|2.2|6.2|84.2|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc_raw|868|121352|94.1|2.1|3.8|2.0|7.9|96.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa|868|74220|96.2|2.6|1.1|4.0|7.7|68.3|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swa_raw|868|76488|90.7|4.0|5.3|5.1|14.4|94.9|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc|868|74644|93.2|4.0|2.8|2.3|9.1|84.2|
|decode_asr_lm_lm_train_lm_transformer_sw_bpe100_valid.loss.ave_asr_model_valid.acc.ave/test_iwslt_swc_raw|868|77530|87.9|5.2|6.9|2.6|14.8|96.7|
