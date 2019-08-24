# v.0.5.0: First multi-speaker Transformer 1024 pt window / 256 pt shift + x-vector with add integration + fast GL 64 iters

## Environments

- date: `Mon Aug 12 20:14:55 JST 2019`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.0`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `d359148ebecf16440526cf0b326dde58960e2d16`
  - Commit date: `Sun Aug 11 23:08:55 2019 +0900`

## Model files

- model link: https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3
- training config file: `conf/train_pytorch_transformer+spkemb.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_clean_460/cmvn.ark`
- e2e file: `exp/train_clean_460_pytorch_train_pytorch_transformer+spkemb/results/model.last1.avg.best`
- e2e JSON file: `exp/train_clean_460_pytorch_train_pytorch_transformer+spkemb/results/model.json`
- dict file: `data/lang_1char/train_clean_460_units.txt`

## Samples

https://drive.google.com/open?id=1_vrdqjM43DdN1Qz7HJkvMQ6lCMmWLeGp


# v.0.3.0: multi-speaker Tacotron2.v1 1024 pt window / 256 pt shift + default taco2 + x-vector + GL 1000 iters

## Environments

- system information: `Linux million5.sp.m.is.nagoya-u.ac.jp 3.10.0-862.14.4.el7.x86_64 #1 SMP Wed Sep 26 15:12:11 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux`
- python version: `Python 3.6.1`
- espnet version: `espnet 0.3.1`
- chainer version: `chainer 5.0.0`
- pytorch version: `pytorch 1.0.0`
- Git hash: `f74a92c7720eb494a10d8e5b0f6a60186a5e741c`

## Model files
- model link: https://drive.google.com/open?id=1iAXwC0AuWusa9AcFeUVkcNLG0I-hnSr3
- training config file: `conf/train_pytorch_tacotron2+spkemb.yaml`
- decoding config file: `conf/decode.yaml`
- cmvn file: `data/train_clean_460/cmvn.ark`
- e2e file: `exp/train_clean_460_pytorch_taco2_r2_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_location128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs64_sort_by_output_mli150_mlo400_sd1/results/model.loss.best`
- e2e JSON file: `exp/train_clean_460_pytorch_taco2_r2_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_location128-15x32_cm_bn_cc_msk_pw1.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs64_sort_by_output_mli150_mlo400_sd1/results/model.json`
- dict file: `data/lang_1char/train_clean_460_units.txt`

## Samples

https://drive.google.com/open?id=18S_B8Ogogij34rIfJOeNF8D--uG7amz2
