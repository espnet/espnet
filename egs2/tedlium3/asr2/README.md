# E-Branchformer, 12 encoder layers, with external language source


## Environments
- date: `Tue Apr 11 01:15:36 EDT 2023`
- python version: `3.8.16 (default, Mar  2 2023, 03:21:46)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.8.1`
- Git hash: `b0cceeac2ecd330e8270789cef945e49058858fa`
  - Commit date: `Thu Mar 30 08:26:54 2023 -0400`


## Model info
- Model link: https://huggingface.co/espnet/dongwei_tedlium3_asr_e-branchformer_external_lm
- ASR config: conf/tuning/train_asr_e_branchformer_size256_mlp1024_e12_mactrue.yaml
- Decode config: conf/tuning/decode_asr.yaml
- LM config: conf/tuning/train_lm_transformer.yaml


## exp/asr_train_asr_e_branchformer_size256_mlp1024_e12_mactrue_raw_en_bpe500_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|27500|94.2|2.5|3.3|0.6|6.4|59.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|145066|96.8|0.5|2.7|0.6|3.8|59.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|54206|95.8|1.6|2.6|0.5|4.7|59.2|

## exp/asr_train_asr_e_branchformer_size256_mlp1024_e12_mactrue_raw_en_bpe500_sp/decode_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|17783|93.6|3.1|3.3|0.9|7.3|69.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|95429|96.5|0.7|2.8|0.8|4.4|69.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|507|36002|95.4|2.0|2.6|0.8|5.5|69.0|




# WIP