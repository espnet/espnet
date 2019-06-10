# Transformer result (default transformer with initial learning rate = 1.0 and epochs = 50)

  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Mon Jun 10 12:34:41 EDT 2019`
    - system information: `Linux b14 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux`
    - python version: `Python 3.7.3`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 6.0.0`
    - pytorch version: `pytorch 1.0.1.post2`
    - Git hash: `82e9b7eb7ccae61e11af28981734ea1c2b315a98`
  - Model files (archived to model.v1.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1BIQBpLRRy3XSMT5IRxnLcgLMirGzu8dg
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - Results (paste them by yourself or obtained by `$ pack_model.sh --results <results>`)
```
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_dev_decode_pytorch_transformer_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |  14326      205341   |   94.1        5.7        0.2        0.1        6.0       42.0   |
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_test_decode_pytorch_transformer_lm/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176      104765   |   93.4        6.4         0.2        0.1        6.7       45.1   |
```

# First result (no tuning, but already very good. cf. Kaldi chain best 7.43% and nnet3 8.64% while ESPnet 8.0%)
```
exp/train_sp_pytorch_no_patience/decode_dev_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.6_rnnlm0.3_2layer_unit650_sgd_bs64/result.txt:
|    SPKR       |     # Snt         # Wrd     |    Corr            Sub           Del           Ins            Err         S.Err     |
|    Sum/Avg    |    14326         205341     |    93.3            6.5           0.2           0.1            6.8          45.2     |
exp/train_sp_pytorch_no_patience/decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.6_rnnlm0.3_2layer_unit650_sgd_bs64/result.txt:
|    SPKR       |     # Snt         # Wrd     |     Corr           Sub            Del           Ins            Err         S.Err     |
|    Sum/Avg    |     7176         104765     |     92.2           7.6            0.2           0.2            8.0          50.2     |
```
