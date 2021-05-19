# RESULTS
## Environments
- date: `Tue Mar  9 09:50:14 EST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.7.1`
- Git hash: `99d89903e42013dda5c5bc08bcf37a529eab7eb7`
  - Commit date: `Tue Mar 9 08:58:35 2021 -0500`

## train_pytorch_train_pytorch_conformer_large_mustc_like_bpe5000_specaug
  - Model files (archived to model.mustc_like.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/107ujDaIrlj6tFHiWLNP6aUBuV0PVyX_Y/view?usp=sharing
    - training config file: `conf/tuning/train_pytorch_conformer_large_mustc_like.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_pytorch_conformer_large_mustc_like_bpe5000_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_pytorch_conformer_large_mustc_like_bpe5000_specaug/results/model.json`
    - dict file: `data/lang_1spm`
  - No LM. 4 GPU training
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_et_librispeech_test_other_decode|2939|71179|92.4|5.7|1.8|1.1|8.7|56.5|
|decode_et_mustc_tst-COMMON_decode|2641|58047|94.7|2.8|2.6|1.1|6.4|36.6|
|decode_et_tedlium2_test_decode|1155|33696|94.1|2.7|3.2|1.2|7.2|56.4|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_et_librispeech_test_other_decode|2939|53022|93.3|6.0|0.7|0.8|7.5|56.4|
|decode_et_mustc_tst-COMMON_decode|2641|47335|95.2|2.9|1.8|1.1|5.8|36.6|
|decode_et_tedlium2_test_decode|1155|27500|94.0|3.0|3.0|1.2|7.2|56.3|

## train_pytorch_train_pytorch_conformer_large_librispeech_like_bpe5000_specaug
  - Model files (archived to model.librispeech_like.tar.gz by `$ pack_model.sh`)
    - model link: https://drive.google.com/file/d/1C2iZQu4P5RKxWAjpD-ZkJZcHIg2-ED51/view?usp=sharing
    - training config file: `conf/tuning/train_pytorch_conformer_large_librispeech_like.yaml`
    - decoding config file: `conf/tuning/decode_pytorch_transformer.yaml`
    - cmvn file: `data/train/cmvn.ark`
    - e2e file: `exp/train_pytorch_train_pytorch_conformer_large_librispeech_like_bpe5000_specaug/results/model.val5.avg.best`
    - e2e JSON file: `exp/train_pytorch_train_pytorch_conformer_large_librispeech_like_bpe5000_specaug/results/model.json`
    - dict file: `data/lang_1spm`
  - No LM. 4 GPU training
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_et_librispeech_test_other_decode|2939|71179|92.7|5.5|1.8|1.0|8.3|54.0|
|decode_et_mustc_tst-COMMON_decode|2641|58047|94.8|2.6|2.6|1.0|6.2|37.0|
|decode_et_tedlium2_test_decode|1155|33696|94.9|2.4|2.6|1.1|6.2|54.3|
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_et_librispeech_test_other_decode|2939|53022|93.7|5.6|0.7|0.8|7.1|53.8|
|decode_et_mustc_tst-COMMON_decode|2641|47335|95.4|2.7|1.8|1.0|5.6|37.0|
|decode_et_tedlium2_test_decode|1155|27500|94.8|2.6|2.5|1.0|6.2|54.3|
