# RESULTS
## Environments
- date: `Tue Mar 23 10:03:49 EDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- pytorch version: `pytorch 1.7.1`
- Git hash: `dcb5bdb2ffa34a9f44255c0b073759c5b9b3f86e`
  - Commit date: `Sat Mar 13 10:16:16 2021 -0500`

## asr_train_asr_raw_en_bpe5000
- https://zenodo.org/record/4630406
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|2043|51075|92.9|4.5|2.6|2.1|9.2|65.6|
|decode_asr_asr_model_valid.acc.ave/test|9627|175116|90.5|7.0|2.5|6.1|15.6|69.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|2043|271188|97.5|0.9|1.6|1.7|4.2|65.6|
|decode_asr_asr_model_valid.acc.ave/test|9627|909930|96.5|1.6|1.9|5.6|9.0|69.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|2043|63598|93.3|3.9|2.8|2.1|8.8|65.6|
|decode_asr_asr_model_valid.acc.ave/test|9627|218851|90.8|6.1|3.1|7.0|16.2|69.3|
