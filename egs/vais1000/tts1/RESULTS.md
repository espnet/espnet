# v.0.5.3: Initial transformer.v1 / fastspeech.v2

This is a demonstration of transformer TTS with VAIS1000 dataset. You should have significantly better model with just adding more data. The VAIS1000 dataset only contain 1000 utterance and total audio length was about one hour. At least 7 hours is needed to get natural, clean and intelligible speech.

- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Fri Oct 25 16:24:36 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `cb8ed9678dce39e239206aff37dab81b2441bdcd`
  - Commit date: `Fri Oct 25 01:39:09 2019 +0900`

## Model

- model link: https://drive.google.com/open?id=1HAG7bF0zlD2SHtIzW4aBX17fzTDONu9X
- training config file: `conf/tuning/train_pytorch_transformer.v1.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_nodev_trim/cmvn.ark`
- e2e file: `exp/train_nodev_trim_pytorch_train_transformer/results/model.last1.avg.best`
- e2e JSON file: `exp/train_nodev_trim_pytorch_train_transformer/results/model.json`
- dict file: `data/lang_1char/train_nodev_trim_units.txt`

## Samples

https://drive.google.com/open?id=1Typ9dRG7jBttY-sXU8y5BUI3eD3W2OJ8

## Model

- model link: https://drive.google.com/open?id=1s4L57xfFQX2mEmNnuoH5w669_MOwup9a
- training config file: `conf/tuning/train_fastspeech.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_nodev_trim/cmvn.ark`
- e2e file: `exp/train_nodev_trim_pytorch_train_fastspeech.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/train_nodev_trim_pytorch_train_fastspeech.v2/results/model.json`
- dict file: `data/lang_1char/train_nodev_trim_units.txt`

## Samples

https://drive.google.com/open?id=1441FvdtdTDyFUu1XomabKB6xat6Ns_Jy
