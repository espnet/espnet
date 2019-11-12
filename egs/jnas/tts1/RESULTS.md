# v.0.5.3 / Initial Transformer

- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments
- date: `Tue Oct 15 00:29:02 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `87a0d491358dea338dd02b0f7c8a5cec4c2f8644`
  - Commit date: `Mon Oct 14 13:16:24 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1fmcr_E_A4bZ2rap4_Aiksg_utlOhXfUJ
- training config file: `conf/train_pytorch_transformer+spkemb.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_nodev_trim/cmvn.ark`
- e2e file: `exp/phn_train_nodev_trim_pytorch_train_pytorch_transformer+spkemb/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_nodev_trim_pytorch_train_pytorch_transformer+spkemb/results/model.json`
- dict file: `data/lang_1char/phn_train_nodev_trim_units.txt`

## Samples

https://drive.google.com/open?id=1fFMQDF6NV5Ysz48QLFYE8fEvbAxCsMBw
