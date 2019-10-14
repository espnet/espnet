# v.0.5.3 Initial Transformer

- FTT in points: 2048
- Shift in points: 256
- Window in points: 1200
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments
- date: `Mon Oct 14 09:59:15 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `36321a262b8dbb6c13297d20e050192624366517`
  - Commit date: `Sat Oct 12 13:24:10 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1bTSygvonv5TS6-iuYsOIUWpN2atGnyhZ
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_phn/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1OgDwp8GVInnqda5ohE_U2DXXrHFFIa17
