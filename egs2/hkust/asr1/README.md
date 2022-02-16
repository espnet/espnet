# RESULTS
## Environments
- date: `Mon Jan 11 08:53:06 JST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.4.0`
- Git hash: `db7dfea809ef919332bd28c392b78378a8df5a77`
  - Commit date: `Mon Jan 4 09:44:09 2021 +0900`

## Transformer ASR + Transformer LM
- ASR: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- LM: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Decode: [conf/tuning/decode.yaml](conf/tuning/decode.yaml)
- Pretrainded model: [https://zenodo.org/record/4430974](https://zenodo.org/record/4430974)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_zh_char_optim_conflr0.001_scheduler_confwarmup_steps25000_batch_bins3000000_accum_grad1_valid.loss.ave_asr_model_valid.acc.ave/dev|5413|56154|81.2|15.1|3.7|2.4|21.2|66.1|

## Transformer ASR w/o LM
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.ave/dev|5413|56154|81.2|15.9|2.8|2.7|21.5|67.1|
