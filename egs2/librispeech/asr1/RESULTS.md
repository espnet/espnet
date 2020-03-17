# Transformer (elayers=12, dlayers=6, units=2048, 1GPU) + LSTM LM (layers=2, units=650, nbpe=500)
- Environments (obtained by `$ get_sys_info.sh`)
    - date: `Sun Mar 15 17:27:58 EDT 2020`
    - system information: `Linux c03 4.9.0-11-amd64 #1 SMP Debian 4.9.189-3+deb9u2 (2019-11-11) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.6.0`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `3ab744011393c2a54a1c334f25fd519193df4468`
- Results
```
exp/asr_transformer_config_fbank_pitch_bpe/decode_dev_cleandecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|    SPKR       |    # Snt         # Wrd     |    Corr           Sub           Del           Ins           Err         S.Err     |
|    Sum/Avg    |    2703          54402     |    95.3           3.9           0.8           0.6           5.3          49.9     |
exp/asr_transformer_config_fbank_pitch_bpe/decode_dev_otherdecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|    SPKR       |    # Snt         # Wrd     |    Corr           Sub           Del           Ins           Err         S.Err     |
|    Sum/Avg    |    2864          50948     |    88.9           9.2           1.9           1.5          12.6          71.4     |
exp/asr_transformer_config_fbank_pitch_bpe/decode_test_cleandecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|    SPKR       |    # Snt         # Wrd     |    Corr            Sub           Del           Ins           Err         S.Err     |
|    Sum/Avg    |    2620          52576     |    95.2            3.9           0.9           0.6           5.4          49.5     |
exp/asr_transformer_config_fbank_pitch_bpe/decode_test_otherdecode_lm_valid.loss.best_asr_model_valid.loss.ave/score_wer/result.txt
|    SPKR       |    # Snt         # Wrd     |    Corr            Sub           Del           Ins           Err         S.Err     |
|    Sum/Avg    |    2939          52343     |    88.7            9.3           2.0           1.5          12.8          72.4     |
```
