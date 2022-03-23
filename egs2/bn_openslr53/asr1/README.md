# RESULTS
## Environments
- date: `Mon Jan 31 10:53:20 EST 2022`
- python version: `3.9.5 (default, Jun  4 2021, 12:28:51)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.6a1`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `9d09bf551a9fe090973de60e15adec1de6b3d054`
  - Commit date: `Fri Jan 21 11:43:15 2022 -0500`
- Pretrained Model: https://huggingface.co/espnet/bn_openslr53

## asr_train_asr_raw_bpe1000
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_lm_lm_train_lm_bpe1000_valid.loss.ave_asr_model_valid.acc.best/sbn_test|2018|6470|74.2|21.3|4.5|2.2|28.0|48.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_lm_lm_train_lm_bpe1000_valid.loss.ave_asr_model_valid.acc.best/sbn_test|2018|39196|89.4|4.3|6.3|1.4|12.0|48.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_lm_lm_train_lm_bpe1000_valid.loss.ave_asr_model_valid.acc.best/sbn_test|2018|15595|77.6|12.7|9.7|1.6|24.0|48.7|

