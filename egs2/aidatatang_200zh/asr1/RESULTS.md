# RESULTS
## Environments
- date: `Tue Jul 21 17:58:50 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `2760b86ef3b7e12f834cd10d56e9482311b61486`
  - Commit date: `Fri Jul 17 11:44:47 2020 -0400`

## asr_train_asr_rnn_fbank_pitch_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|24216|24216|65.3|34.7|0.0|0.0|34.7|34.7|
|decode_test_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|48144|48144|62.8|37.2|0.0|0.0|37.2|37.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|24216|234524|92.3|6.9|0.8|0.2|8.0|34.7|
|decode_test_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|48144|468933|91.4|7.7|0.8|0.3|8.8|37.2|

