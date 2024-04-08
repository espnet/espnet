# E-Branchformer: 16 encoder layers

## Environments
- date: `Wed Feb 22 23:08:40 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `232a317a66eda6c5caee094db4b714bc912dce95`
  - Commit date: `Wed Feb 22 14:22:01 2023 -0600`

## With LM
- ASR config: [conf/tuning/train_asr_e_branchformer_e16_linear1024_lr1e-3.yaml](conf/tuning/train_asr_e_branchformer_e16_linear1024_lr1e-3.yaml)
- Params: 45.43M
- LM config: [conf/train_lm_transformer.yaml](conf/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/aidatatang_200zh_e_branchformer_e16](https://huggingface.co/pyf98/aidatatang_200zh_e_branchformer_e16)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|24216|234524|96.7|2.9|0.4|0.2|3.4|17.6|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|48144|468933|96.1|3.5|0.4|0.2|4.1|20.1|



# E-Branchformer

## Environments
- date: `Mon Dec 26 19:46:01 EST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `7a203d55543df02f0369d5608cd6f3033119a135`
  - Commit date: `Fri Dec 23 00:58:49 2022 +0000`

## asr_train_asr_e_branchformer_linear1024_raw_zh_char_sp
- ASR config: [conf/tuning/train_asr_e_branchformer_linear1024.yaml](conf/tuning/train_asr_e_branchformer_linear1024.yaml)
- Params: 37.66M
- LM config: [conf/train_lm_transformer.yaml](conf/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/aidatatang_200zh_e_branchformer](https://huggingface.co/pyf98/aidatatang_200zh_e_branchformer)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|24216|234524|96.6|3.0|0.4|0.1|3.6|18.4|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|48144|468933|95.9|3.6|0.4|0.2|4.2|20.8|


# Conformer

## Environments
- date: `Fri Dec 24 23:34:58 EST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.7.1`
- Git hash: `a5bacd349a47889aef795f999563018cf201ae64`
  - Commit date: `Wed Dec 22 14:08:29 2021 -0500`

## asr_train_asr_conformer_raw_zh_char_sp
- ASR config: [conf/train_asr_conformer.yaml](conf/train_asr_conformer.yaml)
- Params: 45.98M
- Model link: [https://huggingface.co/sw005320/aidatatang_200zh_conformer](https://huggingface.co/sw005320/aidatatang_200zh_conformer)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|24216|234524|96.6|3.0|0.5|0.1|3.6|18.5|
|decode_asr_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|48144|468933|95.9|3.6|0.4|0.2|4.3|21.0|
