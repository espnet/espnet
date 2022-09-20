# RESULTS

## Environments

- date: `Mon Sep 19 20:16:36 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `30d08e87dfdfdc35ad6f438f01795c3efa531acf`
    - Commit date: `Mon Sep 19 15:31:07 2022 -0400`

## [Conformer](conf/tuning/train_asr_conformer.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer.yaml)

### CER

| dataset                                                                                       | Snt   | Wrd    | Corr | Sub | Del | Ins | Err | S.Err |
|-----------------------------------------------------------------------------------------------|-------|--------|------|-----|-----|-----|-----|-------|
| decode_asr_rnn_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test | 24279 | 243325 | 95.4 | 2.6 | 2.0 | 0.3 | 4.8 | 19.4  |
