# RNN-Transducer (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1KtzW_F4escMuUTvTT41J-Bzdf4tiFBDn
  - training config file: `conf/tuning/transducer/train_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_transducer/results/model.loss.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|85.1|11.6|3.3|3.6|18.5|93.9|
|decode_test_decode_alsd_lm_word7184|760|32771|85.9|11.0|3.2|3.4|17.6|92.1|
|decode_test_decode_default|760|32771|85.0|11.6|3.5|3.4|18.4|93.7|
|decode_test_decode_default_lm_word7184|760|32771|85.6|10.9|3.5|3.2|17.6|92.1|
|decode_test_decode_nsc|760|32771|84.9|11.6|3.5|3.5|18.6|94.1|
|decode_test_decode_nsc_lm_word7184|760|32771|85.7|10.9|3.3|3.3|17.6|92.9|
|decode_test_decode_tsd|760|32771|84.9|11.5|3.6|3.3|18.4|94.3|
|decode_test_decode_tsd_lm_word7184|760|32771|85.7|10.7|3.6|3.0|17.2|91.7|
|decode_train_dev_decode_alsd|100|4007|85.9|11.5|2.5|2.4|16.5|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.6|10.9|2.5|2.2|15.6|97.0|
|decode_train_dev_decode_default|100|4007|85.0|12.0|2.9|2.4|17.4|99.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.9|11.3|2.8|2.2|16.2|96.0|
|decode_train_dev_decode_nsc|100|4007|85.6|11.7|2.8|2.3|16.7|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.6|10.8|2.5|2.2|15.6|97.0|
|decode_train_dev_decode_tsd|100|4007|85.2|11.9|2.9|2.3|17.1|98.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.5|10.5|2.9|2.0|15.4|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|61.7|37.8|0.6|0.6|39.0|93.9|
|decode_test_decode_alsd_lm_word7184|760|7722|64.6|34.8|0.6|0.6|35.9|92.1|
|decode_test_decode_default|760|7722|61.7|37.7|0.6|0.6|38.9|93.7|
|decode_test_decode_default_lm_word7184|760|7722|64.9|34.4|0.8|0.6|35.7|92.1|
|decode_test_decode_nsc|760|7722|61.5|37.9|0.6|0.6|39.1|94.1|
|decode_test_decode_nsc_lm_word7184|760|7722|64.3|35.0|0.7|0.6|36.3|92.9|
|decode_test_decode_tsd|760|7722|61.7|37.7|0.6|0.6|38.9|94.3|
|decode_test_decode_tsd_lm_word7184|760|7722|65.3|34.0|0.7|0.5|35.2|91.7|
|decode_train_dev_decode_alsd|100|927|61.9|38.0|0.1|0.0|38.1|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|65.8|34.1|0.1|0.0|34.2|97.0|
|decode_train_dev_decode_default|100|927|60.3|39.6|0.1|0.0|39.7|99.0|
|decode_train_dev_decode_default_lm_word7184|100|927|64.1|35.8|0.1|0.0|35.9|96.0|
|decode_train_dev_decode_nsc|100|927|61.7|38.2|0.1|0.1|38.4|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|65.0|34.8|0.1|0.1|35.1|97.0|
|decode_train_dev_decode_tsd|100|927|60.9|38.9|0.1|0.0|39.1|98.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|65.8|34.0|0.2|0.0|34.2|96.0|

# Transformer-Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to transformer_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1m-LzNfH6J51zW1-z6D2DLikiWxbUmifX
  - training config file: `conf/tuning/transducer/train_transformer_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_transformer_transducer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_transformer_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|84.6|10.6|4.8|3.5|18.9|95.4|
|decode_test_decode_alsd_lm_word7184|760|32771|85.2|9.6|5.2|3.0|17.8|93.6|
|decode_test_decode_default|760|32771|84.7|10.7|4.6|3.6|18.9|95.5|
|decode_test_decode_default_lm_word7184|760|32771|84.2|9.5|6.3|3.0|18.8|93.6|
|decode_test_decode_nsc|760|32771|84.1|10.8|5.1|3.5|19.4|95.8|
|decode_test_decode_nsc_lm_word7184|760|32771|84.1|9.7|6.3|2.9|18.8|94.2|
|decode_test_decode_tsd|760|32771|83.6|10.7|5.6|3.3|19.6|96.1|
|decode_test_decode_tsd_lm_word7184|760|32771|82.3|9.4|8.3|2.5|20.2|93.9|
|decode_train_dev_decode_alsd|100|4007|85.4|11.5|3.2|2.9|17.5|99.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.3|10.3|3.4|2.6|16.3|98.0|
|decode_train_dev_decode_default|100|4007|84.5|11.7|3.7|2.8|18.3|99.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.3|10.0|4.8|2.6|17.4|98.0|
|decode_train_dev_decode_nsc|100|4007|84.4|11.7|3.9|2.9|18.5|99.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|85.6|9.9|4.4|2.6|17.0|99.0|
|decode_train_dev_decode_tsd|100|4007|83.9|11.4|4.7|2.6|18.7|99.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|84.4|9.4|6.3|2.3|17.9|99.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|58.4|39.5|2.1|0.6|42.2|95.4|
|decode_test_decode_alsd_lm_word7184|760|7722|63.1|34.1|2.8|0.5|37.4|93.6|
|decode_test_decode_default|760|7722|58.3|39.7|2.0|0.6|42.3|95.5|
|decode_test_decode_default_lm_word7184|760|7722|62.2|33.8|4.0|0.5|38.3|93.6|
|decode_test_decode_nsc|760|7722|57.8|39.9|2.3|0.6|42.8|95.8|
|decode_test_decode_nsc_lm_word7184|760|7722|62.0|34.3|3.7|0.5|38.5|94.2|
|decode_test_decode_tsd|760|7722|57.7|39.5|2.7|0.6|42.9|96.1|
|decode_test_decode_tsd_lm_word7184|760|7722|61.3|33.1|5.7|0.4|39.2|93.9|
|decode_train_dev_decode_alsd|100|927|58.6|40.5|1.0|0.1|41.5|99.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|63.9|34.7|1.4|0.0|36.1|98.0|
|decode_train_dev_decode_default|100|927|57.2|41.3|1.5|0.1|42.9|99.0|
|decode_train_dev_decode_default_lm_word7184|100|927|63.5|33.5|2.9|0.0|36.5|98.0|
|decode_train_dev_decode_nsc|100|927|57.1|41.3|1.6|0.1|43.0|99.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|63.5|34.1|2.4|0.1|36.6|99.0|
|decode_train_dev_decode_tsd|100|927|57.2|40.5|2.4|0.1|42.9|99.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|63.1|33.0|3.9|0.1|37.0|99.0|

# Transformer/RNN-Transducer (enc: VGG2L + 8 x Transformer, dec: 1 x LSTM)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to transformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1C-vNmUtWJuy1j27lDumuE5guczbhLyw_
  - training config file: `conf/tuning/transducer/train_transformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_transformer-rnn_transducer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_transformer-rnn_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|85.5|10.2|4.2|2.9|17.3|93.7|
|decode_test_decode_alsd_lm_word7184|760|32771|86.3|9.7|4.0|2.8|16.5|91.6|
|decode_test_decode_default|760|32771|85.6|10.5|4.0|2.9|17.3|93.8|
|decode_test_decode_default_lm_word7184|760|32771|86.5|9.8|3.7|2.8|16.4|91.4|
|decode_test_decode_nsc|760|32771|85.6|10.3|4.1|2.8|17.3|94.5|
|decode_test_decode_nsc_lm_word7184|760|32771|86.3|9.7|3.9|2.7|16.4|92.0|
|decode_test_decode_tsd|760|32771|85.4|10.3|4.3|2.7|17.3|94.1|
|decode_test_decode_tsd_lm_word7184|760|32771|86.0|9.6|4.4|2.4|16.4|91.7|
|decode_train_dev_decode_alsd|100|4007|86.1|11.3|2.5|2.6|16.5|94.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.7|10.8|2.5|2.5|15.8|92.0|
|decode_train_dev_decode_default|100|4007|85.9|11.3|2.8|2.4|16.5|96.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|86.6|10.8|2.6|2.5|15.9|94.0|
|decode_train_dev_decode_nsc|100|4007|86.0|11.2|2.8|2.4|16.4|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.3|10.9|2.8|2.4|16.1|95.0|
|decode_train_dev_decode_tsd|100|4007|86.0|11.0|3.0|2.1|16.1|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.1|10.8|3.1|2.1|16.0|94.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|64.3|34.4|1.2|0.5|36.2|93.7|
|decode_test_decode_alsd_lm_word7184|760|7722|67.0|31.7|1.3|0.5|33.5|91.6|
|decode_test_decode_default|760|7722|64.3|34.7|1.0|0.5|36.2|93.8|
|decode_test_decode_default_lm_word7184|760|7722|67.0|31.9|1.1|0.5|33.5|91.4|
|decode_test_decode_nsc|760|7722|64.4|34.5|1.0|0.6|36.1|94.5|
|decode_test_decode_nsc_lm_word7184|760|7722|66.9|31.9|1.2|0.5|33.6|92.0|
|decode_test_decode_tsd|760|7722|64.4|34.4|1.2|0.5|36.2|94.1|
|decode_test_decode_tsd_lm_word7184|760|7722|66.9|31.6|1.5|0.5|33.6|91.7|
|decode_train_dev_decode_alsd|100|927|64.5|35.2|0.3|0.4|35.9|94.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|67.4|32.1|0.4|0.4|33.0|92.0|
|decode_train_dev_decode_default|100|927|64.8|35.0|0.2|0.3|35.5|96.0|
|decode_train_dev_decode_default_lm_word7184|100|927|67.2|32.5|0.3|0.3|33.1|94.0|
|decode_train_dev_decode_nsc|100|927|64.5|35.3|0.2|0.3|35.8|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|66.3|33.3|0.3|0.3|34.0|95.0|
|decode_train_dev_decode_tsd|100|927|65.0|34.6|0.3|0.3|35.3|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|66.5|33.0|0.5|0.3|33.9|94.0|

# Transformer-Transducer (enc: 3 x TDNN-TDNN-Transformer, dec: 1 x LSTM)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to tdnn_transformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1N5W4Su5WYiAWIl2QPhcy8wsMnLY7SPQM
  - training config file: `conf/tuning/transducer/train_tdnn_transformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_custom_tt/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_custom_tt/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|86.7|9.7|3.7|2.7|16.1|92.4|
|decode_test_decode_alsd_lm_word7184|760|32771|87.2|9.2|3.6|2.6|15.5|90.9|
|decode_test_decode_default|760|32771|86.5|9.8|3.7|2.8|16.2|91.7|
|decode_test_decode_default_lm_word7184|760|32771|87.2|9.1|3.7|2.7|15.5|90.0|
|decode_test_decode_nsc|760|32771|86.6|9.7|3.7|2.7|16.1|92.9|
|decode_test_decode_nsc_lm_word7184|760|32771|87.0|9.4|3.6|2.7|15.7|91.4|
|decode_test_decode_tsd|760|32771|86.6|9.6|3.8|2.6|15.9|92.4|
|decode_test_decode_tsd_lm_word7184|760|32771|87.2|8.9|3.8|2.4|15.1|90.0|
|decode_train_dev_decode_alsd|100|4007|88.5|9.5|2.0|2.4|13.9|91.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|89.4|8.6|2.0|2.4|13.1|87.0|
|decode_train_dev_decode_default|100|4007|88.2|9.6|2.1|2.5|14.3|91.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.0|8.8|2.1|2.5|13.4|87.0|
|decode_train_dev_decode_nsc|100|4007|88.6|9.1|2.3|2.3|13.7|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|88.9|8.7|2.3|2.4|13.5|89.0|
|decode_train_dev_decode_tsd|100|4007|88.6|9.1|2.2|2.2|13.6|92.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|89.6|8.2|2.2|2.0|12.4|87.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|65.9|33.1|1.0|0.6|34.7|92.4|
|decode_test_decode_alsd_lm_word7184|760|7722|68.0|30.9|1.1|0.5|32.5|90.9|
|decode_test_decode_default|760|7722|65.5|33.5|1.0|0.5|35.0|91.7|
|decode_test_decode_default_lm_word7184|760|7722|67.8|31.1|1.1|0.5|32.7|90.0|
|decode_test_decode_nsc|760|7722|65.6|33.3|1.0|0.5|34.9|92.8|
|decode_test_decode_nsc_lm_word7184|760|7722|67.3|31.7|1.0|0.5|33.2|91.3|
|decode_test_decode_tsd|760|7722|66.0|33.0|1.0|0.5|34.5|92.2|
|decode_test_decode_tsd_lm_word7184|760|7722|68.5|30.3|1.2|0.4|31.9|90.0|
|decode_train_dev_decode_alsd|100|927|68.7|31.2|0.1|0.1|31.4|91.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|71.1|28.7|0.2|0.0|28.9|87.0|
|decode_train_dev_decode_default|100|927|68.1|31.7|0.2|0.1|32.0|91.0|
|decode_train_dev_decode_default_lm_word7184|100|927|70.4|29.3|0.2|0.1|29.7|86.0|
|decode_train_dev_decode_nsc|100|927|68.1|31.7|0.2|0.0|31.9|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|69.5|30.3|0.2|0.0|30.5|89.0|
|decode_train_dev_decode_tsd|100|927|68.6|31.2|0.2|0.0|31.4|92.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|71.7|28.0|0.2|0.0|28.3|87.0|

# Conformer-Transducer (enc: VGG2L + 8 x Conformer, dec: 2 x Transformer)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to conformer_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=15AkFzFLM4FTWcfNmt1Ca2fWu2K68wtH-
  - training config file: `conf/tuning/transducer/train_conformer_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_conformer_transducer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_conformer_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|89.3|7.8|3.0|2.1|12.8|88.7|
|decode_test_decode_alsd_lm_word7184|760|32771|90.0|7.2|2.8|2.0|11.9|84.9|
|decode_test_decode_default|760|32771|89.6|7.8|2.6|2.1|12.5|88.0|
|decode_test_decode_default_lm_word7184|760|32771|90.2|7.3|2.4|2.0|11.8|84.5|
|decode_test_decode_nsc|760|32771|88.6|7.9|3.5|2.0|13.5|89.7|
|decode_test_decode_nsc_lm_word7184|760|32771|89.3|7.3|3.4|1.9|12.6|85.8|
|decode_test_decode_tsd|760|32771|88.6|7.9|3.5|1.9|13.4|89.6|
|decode_test_decode_tsd_lm_word7184|760|32771|89.1|7.3|3.6|1.8|12.7|85.7|
|decode_train_dev_decode_alsd|100|4007|88.5|9.5|2.1|1.9|13.4|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|89.2|8.8|2.0|1.8|12.6|88.0|
|decode_train_dev_decode_default|100|4007|88.6|9.4|2.0|1.8|13.1|93.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.1|8.7|2.1|1.7|12.5|87.0|
|decode_train_dev_decode_nsc|100|4007|87.6|9.6|2.8|1.7|14.1|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|88.1|9.0|2.9|1.6|13.6|88.0|
|decode_train_dev_decode_tsd|100|4007|87.5|9.6|2.9|1.7|14.2|93.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|88.1|8.7|3.2|1.5|13.5|87.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|70.2|28.9|0.9|0.3|30.2|88.7|
|decode_test_decode_alsd_lm_word7184|760|7722|73.3|25.8|0.9|0.3|27.1|84.9|
|decode_test_decode_default|760|7722|70.5|28.8|0.6|0.3|29.8|88.0|
|decode_test_decode_default_lm_word7184|760|7722|73.2|26.2|0.6|0.3|27.1|84.5|
|decode_test_decode_nsc|760|7722|69.1|29.8|1.1|0.3|31.2|89.7|
|decode_test_decode_nsc_lm_word7184|760|7722|72.1|26.8|1.1|0.3|28.3|85.8|
|decode_test_decode_tsd|760|7722|69.2|29.8|1.0|0.3|31.1|89.6|
|decode_test_decode_tsd_lm_word7184|760|7722|72.3|26.5|1.2|0.3|28.1|85.7|
|decode_train_dev_decode_alsd|100|927|68.2|31.8|0.0|0.0|31.8|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|70.9|29.1|0.0|0.0|29.1|88.0|
|decode_train_dev_decode_default|100|927|68.3|31.7|0.0|0.0|31.7|93.0|
|decode_train_dev_decode_default_lm_word7184|100|927|70.8|29.0|0.2|0.0|29.2|87.0|
|decode_train_dev_decode_nsc|100|927|66.8|32.9|0.3|0.0|33.2|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|69.5|30.0|0.5|0.0|30.5|88.0|
|decode_train_dev_decode_tsd|100|927|66.7|33.0|0.3|0.0|33.3|93.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|69.8|29.7|0.5|0.0|30.2|87.0|

# Conformer/RNN-Transducer (enc: VGG2L + 8 x Conformer, dec: 1 x LSTM)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to conformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=17-8XfOQAH-6zuRMTZfJ_awojJtchO95l
  - training config file: `conf/tuning/transducer/train_conformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_conformer-rnn_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|89.7|7.4|2.9|2.1|12.4|85.7|
|decode_test_decode_alsd_lm_word7184|760|32771|90.2|7.1|2.7|2.1|11.9|82.5|
|decode_test_decode_default|760|32771|90.1|7.5|2.5|2.2|12.2|85.5|
|decode_test_decode_default_lm_word7184|760|32771|90.5|7.1|2.4|2.1|11.6|82.9|
|decode_test_decode_nsc|760|32771|89.9|7.5|2.6|2.2|12.3|86.3|
|decode_test_decode_nsc_lm_word7184|760|32771|90.4|7.0|2.5|2.1|11.7|83.4|
|decode_test_decode_tsd|760|32771|89.8|7.4|2.8|2.1|12.3|86.1|
|decode_test_decode_tsd_lm_word7184|760|32771|90.2|7.0|2.8|2.0|11.8|83.4|
|decode_train_dev_decode_alsd|100|4007|89.3|8.9|1.8|2.0|12.7|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|89.9|8.3|1.8|1.9|12.0|91.0|
|decode_train_dev_decode_default|100|4007|89.1|8.7|2.1|1.9|12.7|93.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.7|8.4|1.9|1.8|12.2|91.0|
|decode_train_dev_decode_nsc|100|4007|89.5|8.4|2.1|1.8|12.3|93.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|89.8|8.3|1.9|1.9|12.0|93.0|
|decode_train_dev_decode_tsd|100|4007|89.4|8.4|2.2|1.8|12.4|93.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|89.7|8.2|2.1|1.7|11.9|91.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|73.1|26.0|0.9|0.4|27.3|85.7|
|decode_test_decode_alsd_lm_word7184|760|7722|75.0|24.0|0.9|0.4|25.4|82.5|
|decode_test_decode_default|760|7722|73.5|25.9|0.6|0.4|26.9|85.5|
|decode_test_decode_default_lm_word7184|760|7722|75.5|23.8|0.7|0.4|24.9|82.9|
|decode_test_decode_nsc|760|7722|73.3|26.0|0.7|0.4|27.2|86.3|
|decode_test_decode_nsc_lm_word7184|760|7722|75.2|24.1|0.8|0.4|25.2|83.4|
|decode_test_decode_tsd|760|7722|73.3|25.9|0.7|0.4|27.1|86.1|
|decode_test_decode_tsd_lm_word7184|760|7722|75.1|23.9|1.0|0.4|25.3|83.4|
|decode_train_dev_decode_alsd|100|927|69.6|30.4|0.0|0.0|30.4|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|71.8|28.2|0.0|0.0|28.2|91.0|
|decode_train_dev_decode_default|100|927|69.1|30.9|0.0|0.0|30.9|93.0|
|decode_train_dev_decode_default_lm_word7184|100|927|71.3|28.7|0.0|0.0|28.7|91.0|
|decode_train_dev_decode_nsc|100|927|69.8|30.2|0.0|0.0|30.2|93.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|71.3|28.7|0.0|0.0|28.7|93.0|
|decode_train_dev_decode_tsd|100|927|69.7|30.3|0.0|0.0|30.3|93.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|72.3|27.6|0.1|0.0|27.7|91.0|

# Conformer/RNN-Transducer (enc: 3 x TDNN-TDNN-Conformer, dec: 1 x LSTM)

- Environments
  - date: `Thu Nov 19 23:25:08 CET 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.4`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `e9c1a554f0fbeeaeedd0f7e5c9ab096d243011b2`
  - Commit date: `Wed Nov 18 22:06:15 2020 +0100`

- Model files (archived to tdnn_conformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1qUz5UK3rdHRLWPQNuWLCQOmlyrbiUDA3
  - training config file: `conf/tuning/transducer/train_tdnn_conformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_custom_ct/results/model.last10.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_custom_ct/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|90.2|7.4|2.4|2.5|12.2|84.1|
|decode_test_decode_alsd_lm_word7184|760|32771|90.6|7.1|2.3|2.4|11.8|82.0|
|decode_test_decode_default|760|32771|90.2|7.6|2.2|2.4|12.2|83.9|
|decode_test_decode_default_lm_word7184|760|32771|90.7|7.1|2.2|2.2|11.6|81.8|
|decode_test_decode_nsc|760|32771|90.2|7.5|2.3|2.5|12.3|85.5|
|decode_test_decode_nsc_lm_word7184|760|32771|90.6|7.2|2.2|2.4|11.8|84.5|
|decode_test_decode_tsd|760|32771|90.2|7.5|2.3|2.5|12.2|85.1|
|decode_test_decode_tsd_lm_word7184|760|32771|90.6|7.1|2.3|2.2|11.7|82.8|
|decode_train_dev_decode_alsd|100|4007|92.3|6.4|1.2|1.8|9.5|88.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|92.3|6.3|1.3|2.0|9.7|89.0|
|decode_train_dev_decode_default|100|4007|92.2|6.5|1.2|1.8|9.6|85.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|92.4|6.3|1.3|1.7|9.3|84.0|
|decode_train_dev_decode_nsc|100|4007|92.2|6.5|1.3|2.0|9.8|84.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|92.3|6.4|1.3|1.9|9.6|88.0|
|decode_train_dev_decode_tsd|100|4007|92.1|6.5|1.3|2.0|9.8|85.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|92.1|6.3|1.5|1.7|9.6|85.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|73.4|25.9|0.7|0.5|27.1|84.1|
|decode_test_decode_alsd_lm_word7184|760|7722|74.7|24.6|0.7|0.4|25.7|82.0|
|decode_test_decode_default|760|7722|73.3|26.1|0.6|0.4|27.1|83.9|
|decode_test_decode_default_lm_word7184|760|7722|75.1|24.3|0.6|0.4|25.3|81.8|
|decode_test_decode_nsc|760|7722|72.7|26.7|0.6|0.5|27.8|85.4|
|decode_test_decode_nsc_lm_word7184|760|7722|74.3|25.1|0.6|0.5|26.2|84.3|
|decode_test_decode_tsd|760|7722|73.1|26.3|0.6|0.5|27.4|85.1|
|decode_test_decode_tsd_lm_word7184|760|7722|74.8|24.6|0.6|0.4|25.6|82.6|
|decode_train_dev_decode_alsd|100|927|76.9|23.1|0.0|0.1|23.2|88.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|77.2|22.8|0.0|0.1|22.9|89.0|
|decode_train_dev_decode_default|100|927|76.5|23.5|0.0|0.1|23.6|85.0|
|decode_train_dev_decode_default_lm_word7184|100|927|77.5|22.5|0.0|0.1|22.7|84.0|
|decode_train_dev_decode_nsc|100|927|76.2|23.8|0.0|0.5|24.4|84.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|76.8|23.2|0.0|0.4|23.6|88.0|
|decode_train_dev_decode_tsd|100|927|76.5|23.5|0.0|0.5|24.1|85.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|77.5|22.5|0.0|0.5|23.1|85.0|

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