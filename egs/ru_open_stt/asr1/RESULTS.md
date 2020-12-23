# Dataset v1.01, Conformer (4 GPUs, specaug) + Transformer LM (4 GPUs)

  - Model files (archived to espnet-ru-open-stt-20201213-conformer.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1Gj--PtomC5j8pczdJdRMdxInIZsO-jxK/view?usp=sharing
    - training config file: `conf/tuning/train_pytorch_conformer_large.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_pytorch_conformer_large_specaug/results/model.val10.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_pytorch_conformer_large_specaug/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_transformer_large_unigram100/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_transformer_large_unigram100/model.json`
    - dict file: `data/lang_char`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_public_youtube700_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|    SPKR                                     |    # Snt       # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg                                  |    7311        42317    |    82.4         11.8          5.8          1.8         19.4         57.3    |
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_buriy_audiobooks_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|    SPKR                                      |    # Snt       # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg                                   |    7850        39721    |    87.7         10.2          2.0          4.3         16.6         54.0    |
exp/train_pytorch_train_pytorch_conformer_large_specaug/decode_asr_calls_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|    SPKR                               |    # Snt       # Wrd    |    Corr          Sub          Del          Ins          Err        S.Err    |
|    Sum/Avg                            |   12950        79190    |    72.8         18.1          9.1          2.5         29.7         68.6    |
```

# Dataset v1.01, Transformer (4 GPUs, specaug) + Transformer LM (4 GPUs)

  - Model files (archived to espnet-ru-open-stt-20201213-transformer.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1e8ZcH72adBAa7tjU_DMWjaKNjJ89BaUa/view?usp=sharing
    - training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_specaug/results/model.val10.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_specaug/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_transformer_large_unigram100/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_transformer_large_unigram100/model.json`
    - dict file: `data/lang_char`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_pytorch_train_specaug/decode_public_youtube700_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                                    |   # Snt    # Wrd   |  Corr        Sub       Del       Ins        Err     S.Err   |
|  Sum/Avg                                 |   7311     42317   |  81.7       12.1       6.2       1.7       19.9      58.2   |
exp/train_pytorch_train_specaug/decode_buriy_audiobooks_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                                     |   # Snt    # Wrd   |  Corr        Sub       Del       Ins        Err     S.Err   |
|  Sum/Avg                                  |   7850     39721   |  85.8       12.0       2.3       5.6       19.8      62.7   |
exp/train_pytorch_train_specaug/decode_asr_calls_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                              |   # Snt    # Wrd   |  Corr        Sub       Del       Ins        Err     S.Err   |
|  Sum/Avg                           |  12950     79190   |  69.1       17.3      13.7       1.6       32.5      74.1   |
```

# Dataset v0.5, Transformer (4 GPUs, specaug) + LSTM LM (1 GPU)

  - Model files (archived to espnet-ru-open-stt-20190830.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1zQfeMObvxAbVjE8vGyse4b0hz6q-lGMD
    - training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer_large.yaml`
    - fbank file: `conf/fbank.conf`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train/results/model.val10.avg.best`
    - e2e JSON file: `exp/train_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_unigram100/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_unigram100/model.json`
    - dict file: `data/lang_char/train_unigram100_units.txt`
  - Training data format: mono wave with 16000 Hz frequency and 16 bits per channel
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_pytorch_train/decode_public_youtube700_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                                   |  # Snt   # Wrd   |  Corr      Sub      Del      Ins       Err    S.Err  |
|  Sum/Avg                                |  7311    42317   |  81.1     13.1      5.8      1.9      20.8     59.3  |
exp/train_pytorch_train/decode_buriy_audiobooks_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                                    |  # Snt   # Wrd   |  Corr      Sub      Del      Ins       Err    S.Err  |
|  Sum/Avg                                 |  7850    39721   |  86.2     11.5      2.3      4.2      18.1     58.0  |
exp/train_pytorch_train/decode_asr_calls_2_val_model.val10.avg.best_decode_rnnlm.model.best/result.wrd.txt
|  SPKR                             |  # Snt   # Wrd   |  Corr      Sub      Del      Ins       Err    S.Err  |
|  Sum/Avg                          | 12950    79190   |  69.3     18.4     12.3      1.9      32.6     73.2  |
```
