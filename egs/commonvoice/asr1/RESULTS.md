# Catalan results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`

## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1jrMtp3opu6Vq5v9IltCt8p1t-b-OAw0Q
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_ca/cmvn.ark`
    - e2e file: `exp/valid_train_ca_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_ca_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_ca_decode_lm|8607|26845|90.1|6.4|3.5|1.6|11.4|69.6|
|decode_valid_test_ca_decode_lm|8353|26668|89.9|6.5|3.6|1.6|11.6|72.2|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_ca_decode_lm|8607|78013|79.8|17.5|2.6|2.0|22.1|69.5|
|decode_valid_test_ca_decode_lm|8353|76686|79.2|18.2|2.6|2.2|23.0|72.0|
```

# Welsh results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`

## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1mJ1Hc1eklkLJRXrZYG87FzfPO5PHDVoQ
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_cy/cmvn.ark`
    - e2e file: `exp/valid_train_cy_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_cy_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_cy_decode_lm|3395|85111|69.6|19.6|10.8|4.4|34.9|90.4|
|decode_valid_test_cy_decode_lm|3551|89309|67.7|21.3|11.0|5.2|37.6|91.4|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_cy_decode_lm|3395|29842|50.7|38.7|10.6|3.5|52.8|90.3|
|decode_valid_test_cy_decode_lm|3551|30030|48.1|41.8|10.1|4.8|56.7|91.4|
```

# German results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`

## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1VG7h_myFFyLMZPUsNk2Ym-_4VVcyAoTP
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_de/cmvn.ark`
    - e2e file: `exp/valid_train_de_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_de_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_de_decode_lm|29343|826464|89.0|7.5|3.6|1.7|12.8|79.0|
|decode_valid_test_de_decode_lm|28297|803724|89.6|6.9|3.5|1.6|12.0|76.6|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_de_decode_lm|29343|22552|77.1|20.2|2.8|2.9|25.9|78.9|
|decode_valid_test_de_decode_lm|28297|21921|78.1|19.2|2.6|2.8|24.6|76.5|
```

# English results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`

## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1R8H9xYEd82b3B0i5YTXUsja8_KVJ_l4d
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_en/cmvn.ark`
    - e2e file: `exp/valid_train_en_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_en_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_en_decode_lm|66618|181780|86.5|9.0|4.5|2.2|15.7|75.9|
|decode_valid_test_en_decode_lm|60582|160957|86.4|9.1|4.5|2.1|15.7|73.9|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_en_decode_lm|66618|59664|77.5|20.1|2.4|2.2|24.8|75.8|
|decode_valid_test_en_decode_lm|60582|52409|77.9|19.8|2.3|2.5|24.6|73.8|
```

# Spanish results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=13OpZ_Krnv4gR9C_jkynfZioa6y7sj3Ts
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_es/cmvn.ark`
    - e2e file: `exp/valid_train_es_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_es_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_es_decode_lm|2403|58804|79.8|12.5|7.7|3.4|23.6|89.4|
|decode_valid_test_es_decode_lm|2273|54877|79.3|12.7|8.0|3.1|23.8|89.0|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_es_decode_lm|2403|18861|61.3|31.4|7.3|4.6|43.3|89.4|
|decode_valid_test_es_decode_lm|2273|17622|61.0|30.9|8.1|4.4|43.5|89.0|
```

# Basque results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1nt0uAYBHf1LmkJSRaDnT5pjjj859Li-X
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_eu/cmvn.ark`
    - e2e file: `exp/valid_train_eu_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_eu_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_eu_decode_lm|3272|101555|89.7|6.3|4.0|1.3|11.6|79.5|
|decode_valid_test_eu_decode_lm|3292|100387|90.5|5.8|3.7|1.2|10.8|76.8|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_eu_decode_lm|3272|24369|74.5|22.8|2.7|2.3|27.8|79.5|
|decode_valid_test_eu_decode_lm|3292|24217|76.0|21.6|2.4|2.0|26.1|76.6|
```

# Persian results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=10I3Cj9NxkLZyBP2rrGgmvbHpO3Keie6y
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_fa/cmvn.ark`
    - e2e file: `exp/valid_train_fa_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_fa_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_fa_decode_lm|5331|12457|81.9|11.0|7.0|2.4|20.4|87.2|
|decode_valid_test_fa_decode_lm|5376|12642|82.4|10.6|7.0|2.4|20.0|86.6|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_fa_decode_lm|5331|40281|63.2|31.2|5.6|4.5|41.2|87.2|
|decode_valid_test_fa_decode_lm|5376|41030|64.0|30.4|5.6|4.7|40.7|86.5|
```

# French results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1QijqBttQz8EPy_EUVZpvfjCBteBiF4QY
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_fr/cmvn.ark`
    - e2e file: `exp/valid_train_fr_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_fr_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_fr_decode_lm|15227|437256|88.7|7.7|3.6|2.1|13.4|74.4|
|decode_valid_test_fr_decode_lm|15193|432607|88.3|7.9|3.8|2.1|13.8|73.9|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_fr_decode_lm|15227|120352|75.4|21.8|2.8|2.0|26.6|74.3|
|decode_valid_test_fr_decode_lm|15193|119713|75.2|21.8|2.9|2.2|26.9|73.9|
```

# Italian results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=12oV2Tn9D3Q_T2DwUjGXjwx4m2g1xZp6C
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_it/cmvn.ark`
    - e2e file: `exp/valid_train_it_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_it_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_it_decode_lm|2697|98711|84.2|9.7|6.1|2.5|18.3|86.4|
|decode_valid_test_it_decode_lm|2672|96716|83.9|10.0|6.1|2.7|18.8|88.1|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_it_decode_lm|2697|26624|65.3|29.4|5.3|3.5|38.2|86.3|
|decode_valid_test_it_decode_lm|2672|26201|65.1|29.8|5.1|3.7|38.6|87.9|
```

# Kabyle results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`

## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1ylT4WazFn_mCujMnl0DZYKdFtzpSrqew
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_kab/cmvn.ark`
    - e2e file: `exp/valid_train_kab_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_kab_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_kab_decode_lm|17766|356649|87.4|8.2|4.4|2.3|14.9|70.6|
|decode_valid_test_kab_decode_lm|18221|350154|87.7|7.9|4.4|2.3|14.6|69.4|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_kab_decode_lm|17766|96726|70.3|26.1|3.6|2.0|31.8|70.5|
|decode_valid_test_kab_decode_lm|18221|93979|70.8|26.0|3.2|2.5|31.7|69.3|
```

# Russian results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1sC9VT1OuPcjNBsjT2yNtTo0c_A8-6mRo
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_ru/cmvn.ark`
    - e2e file: `exp/valid_train_ru_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_ru_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_ru_decode_lm|1948|85678|83.7|9.9|6.4|2.4|18.7|96.0|
|decode_valid_test_ru_decode_lm|1932|85084|83.5|9.9|6.7|2.4|19.0|96.1|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_ru_decode_lm|1948|16697|52.2|41.7|6.2|3.9|51.7|95.9|
|decode_valid_test_ru_decode_lm|1932|16856|52.0|41.3|6.7|3.7|51.7|96.1|
```

# Tatar results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=131s1Z6GHRFWGr5QFsGQEgQfMjhLHKlyS
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_tt/cmvn.ark`
    - e2e file: `exp/valid_train_tt_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_tt_pytorch_train/results/model.json`
    
### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_tt_decode_lm|2206|58442|88.4|7.2|4.4|1.9|13.5|72.7|
|decode_valid_test_tt_decode_lm|2201|59018|88.4|7.2|4.4|1.9|13.5|73.9|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_tt_decode_lm|2206|12378|68.0|29.1|2.9|2.8|34.8|72.4|
|decode_valid_test_tt_decode_lm|2201|12599|68.3|28.8|2.9|3.2|34.9|73.9|
```

# Chinese (Taiwan) results (default pytorch Transformer setting, 100 epochs, single GPU)
## Environments
- date: `Tue Dec  31 22:52:59 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `a2181ad10929ae980c228f40533defa6904d9db0`
  - Commit date: `Mon Sep 23 22:33:20 2019 +0900`


## Model Information
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/open?id=1hkN1y8d87DN_wiGxFvZwvERd3wKpRTLd
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/valid_train_zh_TW/cmvn.ark`
    - e2e file: `exp/valid_train_zh_TW_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/valid_train_zh_TW_pytorch_train/results/model.json`

### CER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_zh_TW_decode_lm|3979|36127|70.2|22.4|7.3|0.5|30.2|84.2|
|decode_valid_test_zh_TW_decode_lm|4067|36230|70.6|22.1|7.3|0.4|29.8|84.8|
```

### WER
```
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_valid_dev_zh_TW_decode_lm|3979|3979|15.8|84.2|0.0|0.0|84.2|84.2|
|decode_valid_test_zh_TW_decode_lm|4067|4067|15.2|84.8|0.0|0.0|84.8|84.8|
```