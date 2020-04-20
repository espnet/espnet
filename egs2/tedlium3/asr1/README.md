# Transformer (elayers=12, dlayers=6, units=2048, 1GPU) + LSTM LM (layers=2, units=650, nbpe=500)
- Environments (obtained by `$ get_sys_info.sh`)
    - date: `Sun Mar 15 17:27:58 EDT 2020`
    - system information: `Linux c03 4.9.0-11-amd64 #1 SMP Debian 4.9.189-3+deb9u2 (2019-11-11) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.6.0`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `3ab744011393c2a54a1c334f25fd519193df4468`
- Results: Raw feature
```
exp/asr_train_asr_transformer_raw_bpe/decode_dev_decode_lm_train_bpe_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|   SPKR                       |   # Snt       # Wrd    |   Corr         Sub         Del         Ins         Err       S.Err    |
|   Sum/Avg                    |    507        17783    |   88.6         7.6         3.8         2.4        13.8        85.0    |
exp/asr_train_asr_transformer_raw_bpe/decode_test_decode_lm_train_bpe_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|    SPKR                   |   # Snt        # Wrd    |   Corr          Sub         Del          Ins         Err        S.Err    |
|    Sum/Avg                |   1155         27500    |   89.0          6.5         4.5          1.9        12.8         82.9    |
```
- Results: Fbank_pitch feature
```
exp/asr_train_asr_transformer_fbank_pitch_bpe/decode_devdecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|   SPKR                      |   # Snt       # Wrd   |   Corr        Sub         Del        Ins        Err       S.Err   |
|   Sum/Avg                   |    466        14671   |   90.3        6.4         3.3        2.0       11.7        84.1   |
exp/asr_train_asr_transformer_fbank_pitch_bpe/decode_testdecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|   SPKR                   |   # Snt       # Wrd    |   Corr         Sub        Del         Ins         Err       S.Err    |
|   Sum/Avg                |   1144        26651    |   89.6         6.4        4.0         1.9        12.3        82.0    |
```
