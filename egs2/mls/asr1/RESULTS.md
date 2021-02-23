# RESULTS
## Environments
- date: `Fri Jan 22 04:56:26 EST 2021`
- python version: `3.8.3 (default, May 19 2020, 18:47:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.6.0`
- Git hash: `c0c3724fe660abd205dbca9c9bbdffed1d2c79db`
  - Commit date: `Tue Jan 12 23:00:11 2021 -0500`

## asr_transformer_es
- model link: https://zenodo.org/record/4458452
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|88499|81.3|15.6|3.1|2.5|21.2|98.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|474976|94.3|2.9|2.7|1.4|7.1|98.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|251160|88.6|7.9|3.5|2.1|13.6|98.6|
