# Conformer (kernel size=15) + SpecAugment

## MODEL
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1Yj26NM_a2-jkJXQBY2D3_JwUdF8SCcUc/view?usp=sharing
    - training config file: `conf/train.yaml`
    - preprocess config file: `conf/specaug.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train_specuag/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train_specuag/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
    - dict file: `data/lang_1char/train_sp_units.txt

## RESULTS
### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_lm|24216|234524|95.8|3.6|0.5|0.1|4.3|21.8|
|decode_test_decode_lm|48144|468933|95.2|4.3|0.5|0.2|5.0|24.0|


# Conformer result

## MODEL
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1AQdEAvgaYrI625Fa1NPeoDTipjhnvtLP/view?usp=sharing
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
    - dict file: `data/lang_1char/train_sp_units.txt

## RESULTS
### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_lm|24216|234524|94.8|4.6|0.6|0.2|5.4|26.5|
|decode_test_decode_lm|48144|468933|94.1|5.3|0.6|0.2|6.1|28.6|

### Config files
- training config file: conf/tuning/train_pytorch_conformer.yaml
- decoding config file: conf/tuning/decode_pytorch_transformer.yaml


# Transformer result (default transformer with initial learning rate = 2.0 and epochs = 50)

## MODEL
- Model files (archived to model.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1W5oqldNd8yGGyWcUUckJ6zIMDOSLJuPd/view?usp=sharing
    - training config file: `conf/train.yaml`
    - decoding config file: `conf/decode.yaml`
    - cmvn file: `data/train_sp/cmvn.ark`
    - e2e file: `exp/train_sp_pytorch_train/results/model.last10.avg.best`
    - e2e JSON file: `exp/train_sp_pytorch_train/results/model.json`
    - lm file: `exp/train_rnnlm_pytorch_lm/rnnlm.model.best`
    - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
    - dict file: `data/lang_1char/train_sp_units.txt`

## RESULTS
### Environments
- date: `Wed Jul 22 13:36:03 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.5.1`
- Git hash: `949dbbc15cc71783906f079811d5b6b6cee3b119`
  - Commit date: `Wed Jul 22 13:15:26 2020 -0400`

### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_lm|24216|234524|94.3|5.0|0.6|0.2|5.9|28.1|
|decode_test_decode_lm|48144|468933|93.6|5.8|0.6|0.2|6.7|30.6|
