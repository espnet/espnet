# Transformer result (default transformer with initial learning rate = 1.0 and epochs = 18)

  - Environments
    - date: `Mon Jun 29 12:26:49 EDT 2020`
    - system information: `Linux bc-login02 3.10.0-957.1.3.el7_lustre.x86_64 #1 SMP Mon May 27 03:45:37 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.7.0`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
  - Model files (archived to aidatatang_200zh.model.v1.tar.gz)
    - model link: https://drive.google.com/file/d/1gqWx24qRwu-ZS59wczksRGnEADYkjl3m/view?usp=sharing
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results
```
exp/train_sp_pytorch_train/decode_dev_decode_lm/result.txt
| SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg|24216  234524 | 94.1    5.2    0.7    0.2    6.1   27.7 |
exp/train_sp_pytorch_train/decode_test_decode_lm/result.txt
| SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg|48144  468933 | 93.3    6.0    0.7    0.2    6.9   30.0 |
```
