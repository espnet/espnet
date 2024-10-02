# RESULTS
## Environments
- date: `Sat Dec 25 15:43:23 EST 2021`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.9.0`
- Git hash: `cdf0c002f2a64aa6a670cc7675192ac26f0d5add`
  - Commit date: `Fri Dec 24 14:45:33 2021 -0500`

## asr_train_asr_transformer_raw_en_bpe30_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.3_lm0.1/test|130|773|92.9|5.2|1.9|0.3|7.4|31.5|
|beam10_ctc0.3_lm0.1/train_dev|100|591|88.0|8.5|3.6|0.7|12.7|45.0|
|beam10_ctc0.3_lm0/test|130|773|92.8|5.4|1.8|0.4|7.6|30.0|
|beam10_ctc0.3_lm0/train_dev|100|591|88.0|8.0|4.1|0.8|12.9|47.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.3_lm0.1/test|130|2565|96.9|1.0|2.1|0.5|3.6|31.5|
|beam10_ctc0.3_lm0.1/train_dev|100|1915|93.2|2.5|4.3|0.3|7.1|45.0|
|beam10_ctc0.3_lm0/test|130|2565|97.1|1.2|1.8|0.7|3.6|30.0|
|beam10_ctc0.3_lm0/train_dev|100|1915|93.2|2.2|4.6|0.4|7.3|47.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.3_lm0.1/test|130|2695|97.1|1.0|2.0|0.5|3.5|31.5|
|beam10_ctc0.3_lm0.1/train_dev|100|2015|93.5|2.4|4.1|0.2|6.7|45.0|
|beam10_ctc0.3_lm0/test|130|2695|97.2|1.1|1.7|0.7|3.5|30.0|
|beam10_ctc0.3_lm0/train_dev|100|2015|93.5|2.1|4.4|0.4|6.9|47.0|
