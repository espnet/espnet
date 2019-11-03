# v0.5.4: phoneme input / 1200 pt window / 300 pt shift FastSpeech

## Environments

- date: `Wed Oct 30 21:59:31 JST 2019`
- python version: `3.6.7 | packaged by conda-forge | (default, Jul  2 2019, 02:18:42)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `ee0c82d6e74659bd174c90a261135e0901f8b39b`
  - Commit date: `Wed Oct 30 10:44:50 2019 +0900`

## Models

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1CGEDUU9WLVUntC_mkQi7OJzVkGSrj2em
  - training config file: `conf/tuning/train_fastspeech.v1.yaml`
  - decoding config file: `conf/decode.yaml`
  - cmvn file: `data/phn_train_no_dev/cmvn.ark`
  - e2e file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v1/results/model.last1.avg.best`
  - e2e JSON file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v1/results/model.json`
  - dict file: `data/lang_phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1mKutO0eBLH-a99VtP94MeXe-XyqHYrSj

# v0.5.3: phoneme input / 1200 pt window / 300 pt shift + Small Transformer + GL 64 iters

## Environments

- date: `Thu Oct  3 22:54:35 JST 2019`
- python version: `3.6.7 | packaged by conda-forge | (default, Jul  2 2019, 02:18:42)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `ec829fb95dbfcc61672960038643f4f8c8a19480`
  - Commit date: `Sun Sep 29 21:49:36 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD
- training config file: `conf/train_pytorch_transformer.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer_phn/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer_phn/results/model.json`
- dict file: `data/lang_1phn/train_no_dev_units.txt`
- trans_type: phn

## Samples

https://drive.google.com/open?id=1_aTRb5JIw_JWzf0urV0veC5uwhF1abBa

# v0.5.3: phoneme input / 1200 pt window / 300 pt shift + Tacotron 2 + GL 64 iters

## Environments

- date: `Thu Oct  3 22:54:35 JST 2019`
- python version: `3.6.7 | packaged by conda-forge | (default, Jul  2 2019, 02:18:42)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `ec829fb95dbfcc61672960038643f4f8c8a19480`
  - Commit date: `Sun Sep 29 21:49:36 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1kp5M4VvmagDmYckFJa78WGqh1drb_P9t
- training config file: `conf/train_pytorch_tacotron2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2_phn/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2_phn/results/model.json`
- dict file: `data/lang_1phn/train_no_dev_units.txt`
- trans_type: phn

## Samples

https://drive.google.com/open?id=1TZ9MUlzJpxnxgjiAbOe9EHe9PtU2hw4H

# v.0.4.2: 1024 pt window / 512 pt shift + Small Transformer + GL 1000 iters

## Environments

- date: `Thu Aug  1 11:27:37 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.4.2`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `fe5ebe45cd221947005859dcb32642a17c896223`
  - Commit date: `Wed Jul 31 17:10:07 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1VmERSa-yryx5MGNn4EKrbIPBePZNuR03
- training config file: `conf/train_pytorch_transformer.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=13Kwa0FIai_1xOCDgG6xXq9Q6aW24A0BQ

# v.0.3.0: 1024 pt window / 512 pt shift + default taco2 + GL 1000 iters

## Samples

https://drive.google.com/open?id=1QXly67JIkOvwPcZlH5skc37zKH_DyVeG
