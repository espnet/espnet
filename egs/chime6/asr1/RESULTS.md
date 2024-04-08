# Default transducer
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux 5.3.0-42-generic #34~18.04.1-Ubuntu SMP Fri Feb 28 13:42:26 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux`
      - python version: `Python 3.7.3`
      - espnet version: `espnet 0.8.0`
      - chainer version: `chainer 6.0.0`
      - pytorch version: `pytorch 1.0.1.post2`
      - Git hash: `2627f362d99daacc2368386d9cd8ea4727e4ff6a`
  - Model files link `https://drive.google.com/file/d/1vnm7DMZSLprgp4AVLKt5w7jPz4KFy1wQ/view?usp=sharing`
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_worn_simu_u400k_cleaned_trim_sp/cmvn.ark`
    - e2e file: `exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/model.loss.best.ep.10`
    - e2e JSON file: `exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/model.json`
  - Decoding results:
```
CER (or TER) result in exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/decode_dev_gss12.ep.10/result.txt
|  SPKR    |  # Snt    # Wrd   |  Corr      Sub       Del      Ins      Err    S.Err  |
|  Sum/Avg |  7437     280767  |  69.3     13.1      17.7      9.2     39.9     82.4  |
WER result in exp/train_worn_simu_u400k_cleaned_trim_sp_pytorch_train/decode_dev_gss12.ep.10/result.wrd.txt
|  SPKR     |  # Snt    # Wrd   |  Corr       Sub      Del       Ins       Err    S.Err   |
|  Sum/Avg  |  7437     58881   |  52.2      32.4     15.4       7.4      55.3     81.1   |
```

# Default transformer - rnn transducer
  - Environments (obtained by `$ get_sys_info.sh`)
      - system information: `Linux 5.3.0-42-generic #34~18.04.1-Ubuntu SMP Fri Feb 28 13:42:26 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux`
      - python version: `Python 3.7.3`
      - espnet version: `espnet 0.8.0`
      - chainer version: `chainer 6.0.0`
      - pytorch version: `pytorch 1.0.1.post2`
      - Git hash: `2627f362d99daacc2368386d9cd8ea4727e4ff6a`
  - Model files link `https://drive.google.com/file/d/1q5R1_SlyDkhrSnQaufq0qV3MkDFfS3xa/view?usp=sharing`
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yml`
    - cmvn file: `data/train_worn_simu_u400k_cleaned_trim_sp/cmvn.ark`
    - e2e file: `exp/train_worn_simu_u400k_cleaned_trim_nonpos_sp_transformer_rnn_transducer/model.loss.best.ep.10`
    - e2e JSON file: `exp/train_worn_simu_u400k_cleaned_trim_nonpos_sp_transformer_rnn_transducer/model.json`
  - Decoding results:
```
CER (or TER) result in exp/train_worn_simu_u400k_cleaned_trim_nonpos_sp_transformer_rnn_transducer/decode_dev_gss12.ep.10/result.txt
|  SPKR    |  # Snt    # Wrd   |  Corr      Sub       Del      Ins      Err    S.Err  |
|  Sum/Avg |  7437     280767  |  67.3     15.2      17.5     10.9     43.6     81.3  |
WER result in exp/train_worn_simu_u400k_cleaned_trim_nonpos_sp_transformer_rnn_transducer/decode_dev_gss12.ep.10/result.wrd.txt
|  SPKR     |  # Snt    # Wrd   |  Corr       Sub      Del       Ins       Err    S.Err   |
|  Sum/Avg  |  7437     58881   |  48.6      36.7     14.7       8.9      60.3     78.6   |
```
