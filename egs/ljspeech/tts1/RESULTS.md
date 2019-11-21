# v.0.6.0 with frequency limit FastSpeech

- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Thu Nov 21 16:42:08 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `0ad99d180e71e03db9b71c04762a31c0025bab51`
  - Commit date: `Sat Nov 2 21:25:02 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1cDTCWIkiS81YxxRHhTjYTbKoP3cGxxe8
- training config file: `conf/tuning/train_fastspeech.v3.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/char_train_no_dev/cmvn.ark`
- e2e file: `exp/char_train_no_dev_pytorch_train_fastspeech.v3.single/results/model.last1.avg.best`
- e2e JSON file: `exp/char_train_no_dev_pytorch_train_fastspeech.v3.single/results/model.json`
- dict file: `data/lang_1char/char_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1eHppNe-m15yIHadxUxt8bScqBBvSgOeT

## Models

- model link: https://drive.google.com/open?id=1aajckF2Uq1c8g-UhittR7PT-Qa8sgoyl
- training config file: `conf/tuning/train_fastspeech.v3.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v3.single/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v3.single/results/model.json`
- dict file: `data/lang_1phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1wUlnQwFIJeSG7kYHv0NR_U-lvGBX-5HW

# v.0.6.0 with frequency limit Transformer and Tacotron 2

- FTT in points: 1024
- Shift in points: 256
- Frequency limit: 80-7600
- Fast-GL 64 iters

## Environments

- date: `Sat Nov  2 21:25:27 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `0ad99d180e71e03db9b71c04762a31c0025bab51`
  - Commit date: `Sat Nov 2 21:25:02 2019 +0900`

## Models

- model link: https://drive.google.com/open?id=1o_wHmRcspunZUaPV5Q3mQMi-EYG4j6S3
- training config file: `conf/tuning/train_pytorch_tacotron2.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/char_train_no_dev/cmvn.ark`
- e2e file: `exp/char_train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/char_train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.json`
- dict file: `data/lang_1char/char_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1q9Ln5tFb0qNeU65MD3aUShqnKcLGLloo

## Models

- model link: https://drive.google.com/open?id=1Jo06IbVlq79lMA5wM9OMuZ-ByH1eRPkC
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/char_train_no_dev/cmvn.ark`
- e2e file: `exp/char_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/char_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/char_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1Ao_tEZ9f9EQVe9F4r8wcqplFwip1PlWs

## Models

- model link: https://drive.google.com/open?id=17ilwWxjwaHkuXPgYEx1DxuHzpGRfyF4u
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/char_train_no_dev/cmvn.ark`
- e2e file: `exp/char_train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/char_train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1char/char_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1FRjlVrfv-IoYvRweX0AOEVYtJnpA7vjp

## Models

- model link: https://drive.google.com/open?id=1Igiu5AZNz2YL6w8FweiamBK6Tp41nmRK
- training config file: `conf/tuning/train_pytorch_transformer.v3.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/char_train_no_dev/cmvn.ark`
- e2e file: `exp/char_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.last1.avg.best`
- e2e JSON file: `exp/char_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.json`
- dict file: `data/lang_1char/char_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=12O4txCOx1guowz6MSH7iEYjB7ALfcSB0

## Models

- model link: https://drive.google.com/open?id=1pATAby_aO8RY-rLBw22OrBtIal3G49B2
- training config file: `conf/tuning/train_pytorch_tacotron2.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.json`
- dict file: `data/lang_1phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1u_v8kF8YsDPTNVCaDxQ7QSJbD2P6koGx

## Models

- model link: https://drive.google.com/open?id=1lFfeyewyOsxaNO-DEWy9iSz6qB9ZS1UR
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1JFNZapygWsHiP2CXMjTraLzf98h-tEBF

## Models

- model link: https://drive.google.com/open?id=1h-lqoBw0DdOJBlFcPAxyt8cyCcRvwUxO
- training config file: `conf/tuning/train_pytorch_transformer.v1.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v1.single/results/model.json`
- dict file: `data/lang_1phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1Ob743unOppZ2JfWDtBn9mR8oPpzFfqF-

## Models

- model link: https://drive.google.com/open?id=1z8KSOWVBjK-_Ws4RxVN4NTx-Buy03-7c
- training config file: `conf/tuning/train_pytorch_transformer.v3.single.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3.single/results/model.json`
- dict file: `data/lang_1phn/phn_train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1vXcexlhhadvIsZj4ATJYvKbpsIbOuzUU


# v.0.5.3: fastspeech.v3 1024 pt window / 256 pt shift / GL 1000 iters

## Environments

- date: `Wed Oct 16 20:55:50 JST 2019`
- python version: `3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 19:07:31)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `f686d0b1e1e6b0e99215cbb8068df2b33066bdc1`
  - Commit date: `Wed Oct 16 20:45:37 2019 +0900`

# Model files

- model link: https://drive.google.com/open?id=1W86YEQ6KbuUTIvVURLqKtSNqe_eI2GDN
- training config file: `conf/tuning/train_fastspeech.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_fastspeech.v3/results/model.json`
- dict file: `data/lang_1phn/train_no_dev_units.txt`
- trans type: phn

## Samples

https://drive.google.com/open?id=1EHuIHmmb0ft563P-rNaMvsEjRjlR3Jv5

# v.0.5.3: transformer.v3 1024 pt window / 256 pt shift / GL 1000 iters

## Environments

- date: `Wed Oct 16 20:55:50 JST 2019`
- python version: `3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 19:07:31)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.3`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1`
- Git hash: `f686d0b1e1e6b0e99215cbb8068df2b33066bdc1`
  - Commit date: `Wed Oct 16 20:45:37 2019 +0900`

## Model files

- model link: https://drive.google.com/open?id=1M_w7nxI6AfbtSHpMO-exILnAc_aUYvXP
- training config file: `conf/tuning/train_pytorch_transformer.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/phn_train_no_dev/cmvn.ark`
- e2e file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/phn_train_no_dev_pytorch_train_pytorch_transformer.v3/results/model.json`
- dict file: `data/lang_1phn/train_no_dev_units.txt`
- trans type: phn

## Samples

https://drive.google.com/open?id=1UMv2CVfZPlE3o8gOHSflZgGqhS48UsJQ

# v.0.5.0: fastspeech.v2 1024 pt window / 256 pt shift / GL 1000 iters

## Environments

- date: `Wed Jun 26 02:27:43 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `44aa99987ed524dcb5f98421fb3f61df7919ef49`

## Model files

- model link: https://drive.google.com/open?id=1zD-2GMrWM3thaDpS3h3rkTU4jIC0wc5B
- training config file: `conf/tuning/train_fastspeech.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_fastspeech.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_fastspeech.v2/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1PSxs1VauIndwi8d5hJmZlppGRVu2zuy5


# v.0.5.0: fastspeech.v1 1024 pt window / 256 pt shift / GL 1000 iters

## Environments

- date: `Thu Jun 20 10:44:27 JST 2019`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `d57560b16f7dfae5fcaf74ff83e94d9c16dfbe5e`

## Model files

- model link: https://drive.google.com/open?id=17RUNFLP4SSTbGA01xWRJo7RkR876xM0i
- training config file: `conf/tuning/train_fastspeech.v1.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_fastspeech.v1/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_fastspeech.v1/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1tnTQjpz3vrYwnhLufL8iFqVo89lAzdHT


# v.0.4.0: transformer.v2 1024 pt window / 256 pt shift / GL 1000 iters / R=3 / Small

## Environments

- date: `Sun Jun 16 10:03:47 JST 2019`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

## Model files

- model link: https://drive.google.com/open?id=1xxAwPuUph23RnlC5gym7qDM02ZCW9Unp
- training config file: `conf/tuning/train_pytorch_transformer.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v2/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1TqY5cvA2azhl7Xi3E1LFRpsTajlHxO_P


# v.0.4.0: transformer.v1 1024 pt window / 256 pt shift / GL 1000 iters / R=1 / Large

## Environments

- date: `Sun Jun 16 10:03:47 JST 2019`
- system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

## Model files

- model link: https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS
- training config file: `conf/tuning/train_pytorch_transformer.v1.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=14EboYVsMVcAq__dFP1p6lyoZtdobIL1X


# v.0.4.0: tacotron2.v3 1024 pt window / 256 pt shift / GL 1000 iters / R=1 / location-sensitive / guided-attention

## Environments

- date: `Sun Jun 16 10:03:47 JST 2019`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `267da3161cefeae72e9a44bd15e74c0d18591fb6`

## Model files

- model link: https://drive.google.com/open?id=1hiZn14ITUDM1nkn-GkaN_M3oaTOUcn1n
- training config file: `conf/tuning/train_pytorch_tacotron2.v3.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=18JgsOCWiP_JkhONasTplnHS7yaF_konr


# v.0.4.0: tacotron2.v2 1024 pt window / 256 pt shift / GL 1000 iters/ R=1 / forward with transition agent

## Environments

- date: `Fri Jun 14 10:51:01 JST 2019`
- system information: `Linux huracan.sp.m.is.nagoya-u.ac.jp 4.4.0-142-generic #168-Ubuntu SMP Wed Jan 16 21:00:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

## Model files

- model link: https://drive.google.com/open?id=11T9qw8rJlYzUdXvFjkjQjYrp3iGfQ15h
- training config file: `conf/tuning/train_pytorch_tacotron2.v2.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v2/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1cKPDQjLGs7OD8xopSK3YWIGGth37GRSm


# v.0.4.0: tacotron2.v1 1024 pt window / 256 pt shift / GL 1000 iters / R=2 / location-sensitive

## Environments

- date: `Mon Jun 10 10:15:51 JST 2019`
- python version: `Python 3.7.3`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `c86e9311641f59fa397912d0bd2b9c0c599a1127`

## Model files

- model link: https://drive.google.com/open?id=1dKzdaDpOkpx7kWZnvrvx2De7eZEdPHZs
- training config file: `conf/tuning/train_pytorch_tacotron2.v1.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_no_dev/cmvn.ark`
- e2e file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.last1.avg.best`
- e2e JSON file: `exp/train_no_dev_pytorch_train_pytorch_tacotron2.v1/results/model.json`
- dict file: `data/lang_1char/train_no_dev_units.txt`

## Samples

https://drive.google.com/open?id=1ZIDPpb1Bt9V8mrnJCCptMcpIH3SpuyrD


# v.0.3.0: tacotron2 1024 pt window / 256 pt shift + default taco2 + GL 1000 iters

## Samples

https://drive.google.com/open?id=1NclM7_WaoL_Joy71e1bAUfsn_Hcy6HZD
