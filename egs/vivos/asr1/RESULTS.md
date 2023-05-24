# RNN-CTC (4 x BLSTMP)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight1.0|760|32771|80.1|13.0|6.8|2.3|22.2|98.7|
|decode_test_decode_ctcweight1.0_lm|760|32771|84.2|12.0|3.8|3.3|19.1|91.8|
|decode_test_decode_ctcweight1.0_lm_word7184|760|32771|83.0|12.7|4.3|3.2|20.2|93.9|
|decode_train_dev_decode_ctcweight1.0|100|4007|82.6|12.0|5.4|1.7|19.1|99.0|
|decode_train_dev_decode_ctcweight1.0_lm|100|4007|85.3|11.5|3.2|2.1|16.9|93.0|
|decode_train_dev_decode_ctcweight1.0_lm_word7184|100|4007|84.1|12.3|3.5|2.2|18.1|99.0|

# Conformer/Transformer-MTL (enc: Conv2DSubsampling + 8 x Conformer, dec: 2 x Transformer)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to conformer_mtlalpha_0.3.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1sDQXEMrmiCP0HPiLw-Z-q0Av_PdokFiZ
  - training config file: `conf/tuning/train_conformer.yaml`
  - decoding config file: `conf/tuning/decode_ctcweight0.3.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_conformer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_conformer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight0.3|760|32771|89.4|8.1|2.5|2.2|12.9|90.3|
|decode_test_decode_ctcweight0.3_lm_word7184|760|32771|91.5|6.1|2.3|2.0|10.4|77.6|
|decode_train_dev_decode_ctcweight0.3|100|4007|89.8|8.7|1.4|1.8|12.0|94.0|
|decode_train_dev_decode_ctcweight0.3_lm_word7184|100|4007|90.8|7.4|1.8|1.4|10.6|84.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight0.3|760|7722|69.1|30.3|0.6|0.4|31.4|90.3|
|decode_test_decode_ctcweight0.3_lm_word7184|760|7722|78.5|20.6|0.9|0.4|21.9|77.6|
|decode_train_dev_decode_ctcweight0.3|100|927|68.6|31.4|0.0|0.0|31.4|94.0|
|decode_train_dev_decode_ctcweight0.3_lm_word7184|100|927|75.4|23.9|0.6|0.0|24.6|84.0|

# Transducer

## Summary

|Model|Algo|CER¹|WER¹|SER¹|RTF¹²|
|-|-|-|-|-|-|
|RNN-T|default|16.8|38.3|94.1|0.121|
|-|ALSD|16.8|38.4|94.1|0.109|
|-|TSD|16.7|38.2|93.9|0.159|
|-|NSC|16.6|37.9|94.1|0.175|
|-|mAES|16.9|38.5|94.5|0.096|
|RNN-T + Aux|default|15.4|36.7|93.0|0.119|
|-|ALSD|15.4|36.6|93.4|0.109|
|-|TSD|15.4|36.6|93.4|0.159|
|-|NSC|15.5|36.7|93.0|0.176|
|-|mAES|15.5|36.9|93.3|0.095|
|Conformer/RNN-T|default|11.9|26.9|86.1|0.077|
|-|ALSD|12.2|27.3|86.3|0.064|
|-|TSD|12.0|27.0|86.2|0.095|
|-|NSC|12.0|27.0|86.4|0.106|
|-|mAES|12.0|27.1|85.8|0.049|
|Conformer/RNN-T + Aux|default|11.5|26.1|84.5|0.076|
|-|ALSD|11.7|26.3|83.9|0.063|
|-|TSD|11.4|26.0|83.7|0.095|
|-|NSC|11.4|26.0|84.2|0.107|
|-|mAES|11.5|26.3|84.5|0.053|

¹ Reported on the test set only.
² RTF was computed using `line-profiler` tool applied to [recognize method](https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_asr_transducer.py#L470). The reported value is averaged on 5 runs with `nj=1`. All experiments were performed using a single AMD EPYC 7502P.

## RNN-Transducer (Enc: VGG + 4x BLSTM, Dec: 1x LSTM)

- General information
  - GPU: Nvidia A100 40Gb
  - Peak VRAM usage during training: ~ 18.9 GiB
  - Training time: ~ 21 minutes
  - Decoding time (8 jobs, `search-type: default`): ~ 44 seconds

- Environments
  - date: `Sun Aug 15 10:39:18 CEST 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1JkepwVQBJAj-lZxRAyvTqw96pzAOv1Kd
  - training config file: `conf/tuning/transducer/train_rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_rnn_transducer/results/model.loss.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_rnn_transducer/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|86.0|10.9|3.1|2.8|16.8|94.1|
|decode_test_decode_default|760|32771|85.9|10.9|3.2|2.6|16.8|94.1|
|decode_test_decode_maes|760|32771|85.9|10.9|3.2|2.8|16.9|94.5|
|decode_test_decode_nsc|760|32771|86.1|10.7|3.1|2.7|16.6|94.1|
|decode_test_decode_tsd|760|32771|86.0|10.8|3.2|2.7|16.7|93.9|
|decode_train_dev_decode_alsd|100|4007|85.3|12.6|2.1|2.7|17.4|99.0|
|decode_train_dev_decode_default|100|4007|85.1|12.6|2.2|2.5|17.4|98.0|
|decode_train_dev_decode_maes|100|4007|85.3|12.4|2.3|2.6|17.4|98.0|
|decode_train_dev_decode_nsc|100|4007|85.0|12.7|2.3|2.4|17.4|99.0|
|decode_train_dev_decode_tsd|100|4007|84.9|12.8|2.3|2.6|17.7|99.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|62.0|37.3|0.7|0.4|38.4|94.1|
|decode_test_decode_default|760|7722|62.2|37.1|0.7|0.4|38.3|94.1|
|decode_test_decode_maes|760|7722|62.0|37.4|0.6|0.4|38.5|94.5|
|decode_test_decode_nsc|760|7722|62.6|36.7|0.7|0.5|37.9|94.1|
|decode_test_decode_tsd|760|7722|62.2|37.1|0.7|0.4|38.2|93.9|
|decode_train_dev_decode_alsd|100|927|59.7|40.3|0.0|0.1|40.5|99.0|
|decode_train_dev_decode_default|100|927|59.4|40.6|0.0|0.1|40.7|98.0|
|decode_train_dev_decode_maes|100|927|59.7|40.3|0.0|0.1|40.5|98.0|
|decode_train_dev_decode_nsc|100|927|59.1|40.9|0.0|0.0|40.9|99.0|
|decode_train_dev_decode_tsd|100|927|58.9|41.1|0.0|0.0|41.1|99.0|

## RNN-Transducer (Enc: VGG + 4x BLSTM, Dec: 1x LSTM)
##   + CTC loss + Label Smoothing loss

- General information
  - GPU: Nvidia A100 40Gb
  - Training time: ~ 21 minutes
  - Peak memory consumption during training: ~ 18.2 GiB
  - Decoding time (8 jobs, `search-type: default`): ~ 43 seconds

- Environments
  - date: `Sun Aug 15 10:39:18 CEST 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1O16p57K2-Hrg69LNoJGY99I64JFjPWWE
  - training config file: `conf/tuning/transducer/train_rnn_transducer_aux.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_rnn_transducer_aux/results/model.loss.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_rnn_transducer_aux/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|87.1|10.0|2.9|2.5|15.4|93.4|
|decode_test_decode_default|760|32771|87.0|10.0|3.0|2.4|15.4|93.0|
|decode_test_decode_maes|760|32771|87.0|10.1|3.0|2.4|15.5|93.3|
|decode_test_decode_nsc|760|32771|87.0|10.0|3.0|2.5|15.5|93.0|
|decode_test_decode_tsd|760|32771|87.1|9.9|3.0|2.4|15.4|93.4|
|decode_train_dev_decode_alsd|100|4007|87.3|10.8|1.8|2.0|14.6|95.0|
|decode_train_dev_decode_default|100|4007|87.4|10.6|2.0|2.0|14.5|95.0|
|decode_train_dev_decode_maes|100|4007|87.5|10.7|1.9|2.0|14.5|96.0|
|decode_train_dev_decode_nsc|100|4007|87.3|10.7|2.0|1.9|14.6|95.0|
|decode_train_dev_decode_tsd|100|4007|87.5|10.5|2.0|1.8|14.3|95.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|63.9|35.5|0.6|0.4|36.6|93.4|
|decode_test_decode_default|760|7722|63.8|35.5|0.7|0.5|36.7|93.0|
|decode_test_decode_maes|760|7722|63.6|35.8|0.6|0.5|36.9|93.3|
|decode_test_decode_nsc|760|7722|63.8|35.5|0.7|0.5|36.7|93.0|
|decode_test_decode_tsd|760|7722|63.8|35.5|0.7|0.4|36.6|93.4|
|decode_train_dev_decode_alsd|100|927|63.3|36.7|0.0|0.0|36.7|95.0|
|decode_train_dev_decode_default|100|927|63.1|36.9|0.0|0.0|36.9|95.0|
|decode_train_dev_decode_maes|100|927|63.4|36.6|0.0|0.0|36.6|96.0|
|decode_train_dev_decode_nsc|100|927|63.2|36.7|0.1|0.1|36.9|95.0|
|decode_train_dev_decode_tsd|100|927|63.4|36.5|0.1|0.0|36.6|95.0|

## Conformer/RNN-Transducer (Enc: VGG + 8x Conformer, Dec: 1x LSTM)

- General information
  - GPU: Nvidia A100 40Gb
  - Training time: ~ 1 hour
  - Peak memory consumption during training: ~ 11.8 GiB
  - Decoding time (8 job, `search-type: default`): ~ 28 seconds
  - Model averaging: `n_average=10`, `use_valbest_average=true`

- Environments
  - date: `Sun Aug 15 10:39:18 CEST 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1f8nWN76n0iUI1bkRKTJ8wCk2gPF6tHov
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer/results/model.val10.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|89.9|7.5|2.6|2.1|12.2|86.3|
|decode_test_decode_default|760|32771|90.2|7.4|2.3|2.1|11.9|86.1|
|decode_test_decode_maes|760|32771|90.1|7.5|2.4|2.1|12.0|85.8|
|decode_test_decode_nsc|760|32771|90.2|7.5|2.3|2.1|12.0|86.4|
|decode_test_decode_tsd|760|32771|90.1|7.5|2.3|2.1|12.0|86.2|
|decode_train_dev_decode_alsd|100|4007|91.5|7.1|1.3|1.5|10.0|86.0|
|decode_train_dev_decode_default|100|4007|91.2|7.3|1.5|1.5|10.3|90.0|
|decode_train_dev_decode_maes|100|4007|91.2|7.3|1.5|1.5|10.3|89.0|
|decode_train_dev_decode_nsc|100|4007|91.2|7.3|1.5|1.5|10.3|87.0|
|decode_train_dev_decode_tsd|100|4007|91.2|7.3|1.5|1.4|10.3|88.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|73.1|26.2|0.7|0.4|27.3|86.3|
|decode_test_decode_default|760|7722|73.5|26.0|0.5|0.4|26.9|86.1|
|decode_test_decode_default_lm_word7184|760|7722|74.5|24.9|0.6|0.5|25.9|84.6|
|decode_test_decode_maes|760|7722|73.4|26.1|0.5|0.4|27.1|85.8|
|decode_test_decode_nsc|760|7722|73.4|26.0|0.5|0.4|27.0|86.4|
|decode_test_decode_tsd|760|7722|73.4|26.1|0.5|0.4|27.0|86.2|
|decode_train_dev_decode_alsd|100|927|74.6|25.4|0.0|0.0|25.4|86.0|
|decode_train_dev_decode_default|100|927|73.7|26.3|0.0|0.0|26.3|90.0|
|decode_train_dev_decode_default_lm_word7184|100|927|73.8|26.2|0.0|0.0|26.2|87.0|
|decode_train_dev_decode_maes|100|927|73.6|26.4|0.0|0.0|26.4|89.0|
|decode_train_dev_decode_nsc|100|927|73.6|26.4|0.0|0.0|26.4|87.0|
|decode_train_dev_decode_tsd|100|927|73.7|26.3|0.0|0.0|26.3|88.0|

## Conformer/RNN-Transducer (Enc: VGG + 8x Conformer, Dec: 1x LSTM)
##   + CTC loss + Label Smoothing loss

- General information
  - GPU: Nvidia A100 40Gb
  - Training time: ~ 1 hour
  - Peak memory consumption during training: ~ 13.2 GiB
  - Decoding time (8 job, `search-type: default`): ~ 27 seconds
  - Model averaging: `n_average=10`, `use_valbest_average=true`

- Environments
  - date: `Sun Aug 15 10:39:18 CEST 2021`
  - python version: `3.8.5 (default, Sept  4 2020, 07:30:14)  [GCC 7.3.0]`
  - espnet version: `espnet 0.10.2a1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.8.1`
  - Git hash: `4406f25ebf507daf33f68787ff0e3699a0937913`
  - Commit date: `Sat Aug 14 06:25:18 2021 -0400`

- Model files
  - model link: https://drive.google.com/file/d/1y3jZl4vRMK_OoyZOEvFfL6AsbG2gAVBI
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer_aux.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer_aux/results/model.val10.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer_aux/results/model.json`
  - dict file: `data/lang_1char/`

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|90.3|7.1|2.6|2.0|11.7|83.9|
|decode_test_decode_default|760|32771|90.5|7.2|2.3|2.0|11.5|84.5|
|decode_test_decode_maes|760|32771|90.5|7.2|2.3|2.0|11.5|84.5|
|decode_test_decode_nsc|760|32771|90.6|7.1|2.3|2.0|11.4|84.2|
|decode_test_decode_tsd|760|32771|90.5|7.1|2.4|2.0|11.4|83.7|
|decode_train_dev_decode_alsd|100|4007|92.1|6.7|1.2|1.2|9.0|93.0|
|decode_train_dev_decode_default|100|4007|91.7|7.1|1.2|1.4|9.7|94.0|
|decode_train_dev_decode_maes|100|4007|91.7|7.0|1.3|1.3|9.6|93.0|
|decode_train_dev_decode_nsc|100|4007|91.8|6.9|1.3|1.3|9.5|93.0|
|decode_train_dev_decode_tsd|100|4007|91.9|6.9|1.3|1.3|9.4|92.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|74.1|25.3|0.7|0.3|26.3|83.9|
|decode_test_decode_default|760|7722|74.2|25.3|0.5|0.4|26.1|84.5|
|decode_test_decode_maes|760|7722|74.1|25.4|0.5|0.4|26.3|84.5|
|decode_test_decode_nsc|760|7722|74.3|25.2|0.4|0.4|26.0|84.1|
|decode_test_decode_tsd|760|7722|74.3|25.2|0.5|0.4|26.0|83.7|
|decode_train_dev_decode_alsd|100|927|75.6|24.4|0.0|0.0|24.4|93.0|
|decode_train_dev_decode_default|100|927|74.6|25.4|0.0|0.0|25.4|94.0|
|decode_train_dev_decode_maes|100|927|74.4|25.6|0.0|0.0|25.6|93.0|
|decode_train_dev_decode_nsc|100|927|74.6|25.4|0.0|0.0|25.4|93.0|
|decode_train_dev_decode_tsd|100|927|74.6|25.4|0.0|0.0|25.4|92.0|
