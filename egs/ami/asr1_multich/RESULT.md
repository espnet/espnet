  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Wed Jun 26 07:08:45 EDT 2019`
    - system information: `Linux b15 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.6.8 :: Anaconda, Inc.
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.0`
    - Git hash: `7c81e64a37af9b7593b43faa95d8b9833f90ecad`

  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: 
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - e2e file: `exp/mdm8_train_pytorch_data_augment/results/model.acc.best`
    - e2e JSON file: `exp/mdm8_train_pytorch_data_augment/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm_word20000/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm_word20000/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/mdm8_train_pytorch_data_augment/decode_mdm8_dev_decode_lm_word20000/result.wrd.txt
| SPKR                | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg             |13059  94914 | 63.1   29.8    7.1    8.1   45.0   79.4 |
exp/mdm8_train_pytorch_data_augment/decode_mdm8_eval_decode_lm_word20000/result.wrd.txt
| SPKR                 | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
| Sum/Avg              |12612  89635 | 58.6   32.4    9.0    7.8   49.2   78.1 |
```