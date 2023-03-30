# Conformer, 12 encoder layers, with external language source

## Environments
- date: `Mon Mar 27 04:02:03 EDT 2023`
- python version: `3.8.16 (default, Mar  2 2023, 03:21:46)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.8.1`
- Git hash: `ff841366229d539eb74d23ac999cae7c0cc62cad`
  - Commit date: `Mon Feb 20 12:23:15 2023 -0500`

## exp/asr_train_raw_en_bpe500_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|14671|94.0|2.7|3.3|0.7|6.6|65.9|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|27500|93.9|2.7|3.4|0.7|6.8|61.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|78259|96.6|0.6|2.8|0.6|4.0|65.9|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|145066|96.6|0.6|2.8|0.6|4.1|61.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|29364|95.5|1.9|2.7|0.5|5.1|65.9|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|54206|95.5|1.7|2.7|0.6|5.1|61.1|
