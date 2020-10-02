# Conformer result with specaug, speed perturbation, and RNNLM
## Environments
- date: `Fri Oct  2 13:38:36 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `ac7510ea0ebf298c9bd6ff6754f6dba872eaad1f`
  - Commit date: `Fri Sep 25 11:57:31 2020 +0000`

## Model
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1bacBj-kpR6N7dYnJnWrD22iBVZhaRfx-
  - training config file: `conf/train.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/tr_no_dev_sp/cmvn.ark`
  - e2e file: `exp/tr_no_dev_sp_pytorch_train_pytorch_conformer.lr10.ep50_specaug/results/model.last10.avg.best`
  - e2e JSON file: `exp/tr_no_dev_sp_pytorch_train_pytorch_conformer.lr10.ep50_specaug/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`

## tr_no_dev_sp_pytorch_train_pytorch_conformer.lr10.ep50_specaug
### CER (48k)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_pytorch_transformer|250|6349|88.2|9.6|2.1|1.2|13.0|86.0|
|decode_eval1_decode_pytorch_transformer|250|5928|86.7|10.7|2.5|1.3|14.5|88.0|

# RNN model(elayer=4, units=1024)
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Fri Jun 14 12:35:38 JST 2019`
    - system information: `Linux chikaku1.sp.m.is.nagoya-u.ac.jp 3.10.0-862.14.4.el7.x86_64 #1 SMP Wed Sep 26 15:12:11 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux`
    - python version: `Python 3.6.5`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 5.0.0`
    - pytorch version: `pytorch 1.0.0`
    - Git hash: `1e56f57bfb57cd49988e4a62417ab27bf2a0013b`
  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link:https://drive.google.com/open?id=1dboEtdxanufX3w0zVcX2B2JhBmbrsHHO
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/tr_no_dev/cmvn.ark`
    - e2e file: `exp/tr_no_dev_pytorch_train/results/model.acc.best`
    - e2e JSON file: `exp/tr_no_dev_pytorch_train/results/model.json`
  - Results

```
# 16k
write a CER (or TER) result in exp/tr_no_dev_pytorch_train/decode_eval1_decode/result.txt
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      | Sum/Avg|  250    5928 | 79.2   18.7    2.2    3.8   24.7   98.4 |

# 48k
write a CER (or TER) result in exp/tr_no_dev_pytorch_train/decode_eval1_decode/result.txt
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      | Sum/Avg|  250    5928 | 81.8   15.7    2.5    2.4   20.6   97.2 |

```
# Transformer (v1 model)
  - Environments (obtained by `$ get_sys_info.sh`)
    - date: `Fri Jun 14 16:04:00 JST 2019`
    - system information: `Linux chikaku1.sp.m.is.nagoya-u.ac.jp 3.10.0-862.14.4.el7.x86_64 #1 SMP Wed Sep 26 15:12:11 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux`
    - python version: `Python 3.6.5`
    - espnet version: `espnet 0.3.1`
    - chainer version: `chainer 5.0.0`
    - pytorch version: `pytorch 1.0.0`
    - Git hash: `00b87ac02e9b3ba759b22ccef359aa77a356e347`

  - Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1dboEtdxanufX3w0zVcX2B2JhBmbrsHHO
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/tr_no_dev/cmvn.ark`
    - e2e file: `exp/tr_no_dev_pytorch_train/results/model.acc.best`
    - e2e JSON file: `exp/tr_no_dev_pytorch_train/results/model.json`

```
#16k transformer
write a CER (or TER) result in exp/tr_no_dev_pytorch_train/decode_eval1_decode/result.txt
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      | Sum/Avg|  250    5928 | 80.6   15.2    4.2    1.9   21.3   97.6 |

# 48k transformer
write a CER (or TER) result in exp/tr_no_dev_pytorch_train/decode_eval1_decode/result.txt
      | SPKR   | # Snt  # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
      | Sum/Avg|  250    5928 | 83.1   13.6    3.3    1.8   18.7   94.8 |

```
