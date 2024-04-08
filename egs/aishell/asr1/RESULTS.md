# Conformer-Transducer with auxiliary task (CTC weight = 0.5)

## Environments
- Same as RNN-Transducer (see below)

## Config files
- preprocess config: `conf/specaug.yaml`
- train config: `conf/tuning/transducer/train_conformer-rnn_transducer_aux_ngpu4.yaml`
- lm config: `-` (LM was not used)
- decode config: `conf/tuning/transducer/decode_default.yaml`
- ngpu: `4`

## Results (CER)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_default|14326|205341|95.8|4.0|0.2|0.1|4.3|33.6|
|decode_test_decode_default|7176|104765|95.3|4.4|0.2|0.1|4.8|36.3|


# Conformer-Transducer

## Environments
- Same as RNN-Transducer (see below)

## Config files
- preprocess config: `conf/specaug.yaml`
- train config: `conf/tuning/transducer/train_conformer-rnn_transducer.yaml`
- lm config: `-` (LM was not used)
- decode config: `conf/tuning/transducer/decode_default.yaml`

## Results (CER)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_default|14326|205341|95.6|4.2|0.2|0.1|4.5|34.0|
|decode_test_decode_default|7176|104765|95.0|4.7|0.3|0.1|5.0|37.1|


# RNN-Transducer with auxiliary task (CTC weight = 0.1)

## Environments
- Same as RNN-Transducer (see below)

## Config files
- preprocess config: `conf/specaug.yaml`
- train config: `conf/tuning/transducer/train_transducer_aux.yaml`
- lm config: `-` (LM was not used)
- decode config: `conf/tuning/transducer/decode_default.yaml`

## Results (CER)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_default|14326|205341|93.9|5.8|0.3|0.1|6.3|41.9|
|decode_test_decode_default|7176|104765|93.2|6.5|0.4|0.1|6.9|44.5|


# RNN-Transducer

## Environments
- date: `Thu May 20 05:29:03 UTC 2021`
- python version: `3.7.4 (default, Aug 13 2019, 20:35:49)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `95b3008cdcc2247e781a048bc999243dc7f45fe7`
  - Commit date: `Sat Mar 6 00:48:29 2021 +0000`

## Config files
- preprocess config: `conf/specaug.yaml`
- train config: `conf/tuning/transducer/train_transducer.yaml`
- lm config: `-` (LM was not used)
- decode config: `conf/tuning/transducer/decode_default.yaml`

## Results (CER)
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_default|14326|205341|93.8|5.9|0.3|0.1|6.3|42.0|
|decode_test_decode_default|7176|104765|92.9|6.7|0.3|0.1|7.2|45.9|


# Conformer (kernel size = 15) + SpecAugment + LM weight = 0.0 result

- training config file: `conf/tuning/train_pytorch_conformer_kernel15.yaml`
- preprocess config file: `conf/specaug.yaml`
- decoding config file: `conf/decode.yaml`, set `lm-weight = 0.0`
- model link: https://drive.google.com/file/d/1pOhwj6JFqVyt5quW7BKWfJ3vfPFRoxpQ/view?usp=sharing
```
exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/decode_dev_decode_lm0.0/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |  14326      205341   |   95.4        4.5        0.1        0.1        4.6       36.0   |
exp/train_sp_pytorch_train_pytorch_conformer_kernel15_specaug/decode_test_decode_lm0.0/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176      104765   |   95.0        4.9         0.1        0.1        5.1       38.6   |
```

# Conformer (kernel size = 31) + SpecAugment + LM weight = 0.0 result

- training config file: `conf/tuning/train_pytorch_conformer_kernel31.yaml`
- preprocess config file: `conf/specaug.yaml`
- decoding config file: `conf/decode.yaml`, set `lm-weight = 0.0`
```
exp/train_sp_pytorch_train_pytorch_conformer_kernel31_specaug/decode_dev_decode_lm0.0/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |  14326      205341   |   95.4        4.5        0.1        0.1        4.7       36.2   |
exp/train_sp_pytorch_train_pytorch_conformer_kernel31_specaug/decode_test_decode_lm0.0/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176      104765   |   94.9        5.0         0.1        0.1        5.2       39.1   |
```

# Conformer (kernel size = 31) result

- training config file: `conf/tuning/train_pytorch_conformer_kernel31.yaml`
- decoding config file: `conf/decode.yaml`
```
exp/train_sp_pytorch_train_pytorch_conformer_kernel31/decode_dev_decode/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |  14326      205341   |   94.9        5.0        0.1        0.1        5.2       38.3   |
exp/train_sp_pytorch_train_pytorch_conformer_kernel31/decode_test_decode/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub         Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176      104765   |   94.2        5.4         0.2        0.1        5.8       41.0   |
```

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

# Ngram related
   - decoding with ngram and RNNLM
```
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_dev_decode_pytorch_transformer_lm0.7_4gramfull0.3/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   14326      205341  |   94.1        5.7        0.2        0.1        6.0      41.7    |
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_test_decode_pytorch_transformer_lm0.7_4gramfull0.3/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176       104765  |   93.5        6.3        0.2        0.1        6.6      44.6    |
```
```
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_dev_decode_pytorch_transformer_lm0.7_4grampart0.3/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   14326      205341  |   94.1        5.7        0.2        0.1        6.0      41.7    |
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_test_decode_pytorch_transformer_lm0.7_4grampart0.3/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176       104765  |   93.5        6.3        0.2        0.1        6.6      44.6    |
```
  - only e2e model
```
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_dev_decode_pytorch_transformer/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   14326       205341 |   93.6        6.2        0.2        0.1        6.5      45.6    |
exp/train_sp_pytorch_train_pytorch_transformer_lr1.0/decode_test_decode_pytorch_transformer/result.txt
|   SPKR     |   # Snt      # Wrd   |   Corr        Sub        Del        Ins        Err      S.Err   |
|   Sum/Avg  |   7176       104765  |   92.7        7.1        0.2        0.1        7.4      49.8    |
```
