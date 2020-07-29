# Transformer (elayers=12, dlayers=6, units=2048, 4 GPUs, specaug)

  - Model files (archived to espnet-ru-open-stt-20190830.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1zQfeMObvxAbVjE8vGyse4b0hz6q-lGMD
    - training config file: `conf/tuning/train_pytorch_transformer_large_ngpu4.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer_large.yaml`
    - fbank file: `/conf/fbank.conf`
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
