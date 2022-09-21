# RESULTS

## Environments

- date: `Wed Sep 21 01:11:58 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `9d0f3b3e1be6650d38cc5008518f445308fe06d9`
    - Commit date: `Mon Sep 19 20:27:41 2022 -0400`

## [Conformer](conf/tuning/train_asr_conformer.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer.yaml)

### CER

| dataset                                                                                       | Snt   | Wrd    | Corr | Sub | Del | Ins | Err | S.Err |
|-----------------------------------------------------------------------------------------------|-------|--------|------|-----|-----|-----|-----|-------|
| decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test | 24279 | 243325 | 96.4 | 1.7 | 2.0 | 0.1 | 3.7 | 15.6  |