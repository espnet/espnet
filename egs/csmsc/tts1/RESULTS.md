# v.0.5.3 FastSpeech

- FTT in points: 2048
- Shift in points: 300
- Window in points: 1200
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Tue Oct 22 11:05:33 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `879821d4ff495e677d963873da0345497fafd29a`
  - Commit date: `Sun Oct 20 23:25:04 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1T8thxkAxjGFPXPWPTcKLvHnd6lG0-82R
- training config file: `conf/tuning/train_fastspeech.v3.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_fastspeech.v3.single/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_fastspeech.v3.single/results/model.json`
- dict file: `data/lang_phn/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1Ol_048Tuy6BgvYm1RpjhOX4HfhUeBqdK

# v.0.5.3 Initial Transformer

- FTT in points: 2048
- Shift in points: 300
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
