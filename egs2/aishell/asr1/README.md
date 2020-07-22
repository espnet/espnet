## Environments
- date: `Sun Feb  2 02:03:55 CST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- pytorch version: `pytorch 1.1.0`
- Git hash: `e0fd073a70bcded6a0e6a3587630410a994ccdb8` (+ fixing https://github.com/espnet/espnet/pull/1533)
  - Commit date: `Sat Jan 11 06:09:24 2020 +0900`

## asr_train_asr_rnn_new_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_devdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|14326|205341|92.6|7.2|0.2|0.1|7.5|49.6|
|decode_testdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|7176|104765|91.6|8.2|0.3|0.2|8.6|53.4|

## asr_train_asr_transformer_lr0.002_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|14326|205341|93.3|6.5|0.2|0.1|6.8|45.6|
|decode_test_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|7176|104765|92.7|7.1|0.3|0.1|7.4|47.6|

