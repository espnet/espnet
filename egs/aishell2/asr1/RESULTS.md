# Transformer result (default transformer with initial learning rate = 1.0 and epochs = 25)

  - Environments
    - date: `Tue Feb 25 16:18:51 EST 2020`
    - system information: `Linux node9 4.15.0-72-generic #81~16.04.1-Ubuntu SMP Tue Nov 26 16:34:21 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
    - python version: `Python 3.6.8`
    - espnet version: `espnet 0.6.2`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `9ec6939a580c4f8ab6cefb9cb13614e44e72a627`
  - Model files (archived to model.v1.tar.gz)
    - model link: https://drive.google.com/drive/folders/1qsPhnQJOahBHcNmA4EnWwCBNp2oiZivC
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results
```
exp/train_sp_pytorch_train/decode_test_android_avg_best_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   5000       49534   |   91.2        8.4        0.4        0.1        8.9       46.4   |
exp/train_sp_pytorch_train/decode_test_ios_avg_best_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   5000       49534   |   92.6        7.1        0.3        0.1        7.5       41.7   |
exp/train_sp_pytorch_train/decode_test_mic_avg_best_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   5000       49534   |   91.6        8.1        0.3        0.1        8.6       45.3   |
exp/train_sp_pytorch_train/decode_test_overall_avg_best_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   15000      148602  |   91.8        7.9        0.3        0.1        8.3       44.5   |
```
