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

# RNN-Transducer w/ att. (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to rnn_transducer_att.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1dmH07AxYYmJq1mLhMxPfRHl1R_LqaY79
  - training config file: `conf/tuning/transducer/train_transducer_att.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_transducer_att/results/model.loss.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_transducer_att/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|84.7|11.9|3.4|3.5|18.8|94.7|
|decode_test_decode_alsd_lm_word7184|760|32771|85.4|11.3|3.3|3.2|17.8|93.0|
|decode_test_decode_default|760|32771|84.5|11.9|3.6|3.2|18.7|94.7|
|decode_test_decode_default_lm_word7184|760|32771|85.2|11.2|3.6|3.0|17.8|92.5|
|decode_test_decode_nsc|760|32771|84.5|11.9|3.6|3.3|18.9|94.9|
|decode_test_decode_nsc_lm_word7184|760|32771|85.3|11.2|3.5|3.1|17.8|93.4|
|decode_test_decode_tsd|760|32771|84.5|11.8|3.7|3.1|18.6|94.2|
|decode_test_decode_tsd_lm_word7184|760|32771|85.1|11.0|3.9|2.7|17.6|92.6|
|decode_train_dev_decode_alsd|100|4007|86.8|10.8|2.3|2.4|15.5|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|87.1|10.4|2.5|2.3|15.2|94.0|
|decode_train_dev_decode_default|100|4007|86.2|11.0|2.8|2.1|15.8|98.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|86.7|10.5|2.8|2.1|15.4|95.0|
|decode_train_dev_decode_nsc|100|4007|86.0|11.1|2.8|2.2|16.2|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.8|10.4|2.8|2.2|15.4|96.0|
|decode_train_dev_decode_tsd|100|4007|86.2|11.0|2.8|2.0|15.7|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.6|10.2|3.1|1.9|15.2|95.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|60.7|38.7|0.6|0.5|39.8|94.7|
|decode_test_decode_alsd_lm_word7184|760|7722|63.7|35.6|0.7|0.5|36.8|93.0|
|decode_test_decode_default|760|7722|60.4|39.0|0.6|0.5|40.1|94.7|
|decode_test_decode_default_lm_word7184|760|7722|63.4|35.8|0.8|0.4|37.0|92.5|
|decode_test_decode_nsc|760|7722|60.2|39.1|0.7|0.5|40.3|94.9|
|decode_test_decode_nsc_lm_word7184|760|7722|63.2|36.1|0.7|0.5|37.3|93.4|
|decode_test_decode_tsd|760|7722|60.6|38.7|0.7|0.5|39.9|94.2|
|decode_test_decode_tsd_lm_word7184|760|7722|63.9|35.1|1.0|0.4|36.5|92.6|
|decode_train_dev_decode_alsd|100|927|64.0|36.0|0.0|0.1|36.1|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|66.1|33.9|0.0|0.1|34.0|94.0|
|decode_train_dev_decode_default|100|927|63.3|36.6|0.1|0.0|36.7|98.0|
|decode_train_dev_decode_default_lm_word7184|100|927|65.6|34.3|0.1|0.0|34.4|95.0|
|decode_train_dev_decode_nsc|100|927|62.7|37.2|0.1|0.0|37.3|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|65.3|34.5|0.2|0.0|34.7|96.0|
|decode_train_dev_decode_tsd|100|927|63.3|36.7|0.0|0.0|36.7|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|66.1|33.7|0.2|0.0|33.9|95.0|

# Transformer-Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to transformer_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1qGGTCbZGGTbkj3A22C-ornRw5Gsf7rLy
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
|decode_test_decode_alsd|760|32771|83.9|10.8|5.3|3.0|19.1|96.3|
|decode_test_decode_alsd_lm_word7184|760|32771|84.5|9.5|6.0|2.5|18.0|93.8|
|decode_test_decode_default|760|32771|83.9|10.9|5.1|3.1|19.2|96.8|
|decode_test_decode_default_lm_word7184|760|32771|83.8|9.6|6.6|2.7|18.9|93.7|
|decode_test_decode_nsc|760|32771|83.8|10.8|5.4|3.0|19.2|96.6|
|decode_test_decode_nsc_lm_word7184|760|32771|83.8|9.6|6.6|2.6|18.8|94.3|
|decode_test_decode_tsd|760|32771|83.2|10.8|6.0|2.8|19.7|96.7|
|decode_test_decode_tsd_lm_word7184|760|32771|82.2|9.3|8.5|2.3|20.0|94.2|
|decode_train_dev_decode_alsd|100|4007|85.2|11.6|3.3|2.3|17.1|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|85.5|10.6|3.9|1.9|16.4|97.0|
|decode_train_dev_decode_default|100|4007|85.9|11.3|2.7|2.4|16.5|98.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.4|10.5|4.1|2.1|16.7|96.0|
|decode_train_dev_decode_nsc|100|4007|85.3|11.5|3.2|2.3|17.0|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|85.6|10.6|3.8|2.1|16.4|96.0|
|decode_train_dev_decode_tsd|100|4007|84.9|11.5|3.6|2.3|17.4|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|83.0|10.0|7.0|1.6|18.6|97.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|57.8|39.9|2.3|0.5|42.7|96.3|
|decode_test_decode_alsd_lm_word7184|760|7722|62.4|34.2|3.3|0.4|38.0|93.8|
|decode_test_decode_default|760|7722|57.6|40.1|2.3|0.6|43.0|96.8|
|decode_test_decode_default_lm_word7184|760|7722|61.7|34.2|4.0|0.5|38.8|93.7|
|decode_test_decode_nsc|760|7722|57.5|40.1|2.4|0.5|43.0|96.6|
|decode_test_decode_nsc_lm_word7184|760|7722|61.6|34.5|3.9|0.5|38.9|94.3|
|decode_test_decode_tsd|760|7722|57.3|39.9|2.8|0.5|43.2|96.7|
|decode_test_decode_tsd_lm_word7184|760|7722|60.9|33.5|5.6|0.4|39.5|94.2|
|decode_train_dev_decode_alsd|100|927|58.0|40.7|1.3|0.1|42.1|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|63.4|34.7|1.8|0.1|36.7|97.0|
|decode_train_dev_decode_default|100|927|59.1|40.0|0.9|0.1|41.0|98.0|
|decode_train_dev_decode_default_lm_word7184|100|927|62.8|34.7|2.5|0.1|37.3|96.0|
|decode_train_dev_decode_nsc|100|927|58.4|40.5|1.2|0.1|41.7|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|63.6|34.4|1.9|0.1|36.5|96.0|
|decode_train_dev_decode_tsd|100|927|57.8|40.6|1.6|0.1|42.3|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|62.7|32.5|4.9|0.1|37.4|97.0|

# Transformer/RNN-Transducer (enc: VGG2L + 6 x Transformer, dec: 1 x LSTM)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to transformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1DbxunJLXFCczYf1BuSserirhAOTY8qmo
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
|decode_test_decode_alsd|760|32771|86.1|10.2|3.7|3.2|17.1|93.4|
|decode_test_decode_alsd_lm_word7184|760|32771|86.9|9.5|3.6|3.0|16.1|91.7|
|decode_test_decode_default|760|32771|86.1|10.4|3.5|3.2|17.1|93.3|
|decode_test_decode_default_lm_word7184|760|32771|87.0|9.8|3.2|3.1|16.0|91.7|
|decode_test_decode_nsc|760|32771|86.1|10.3|3.6|3.1|17.0|93.6|
|decode_test_decode_nsc_lm_word7184|760|32771|86.9|9.7|3.4|2.9|16.0|92.2|
|decode_test_decode_tsd|760|32771|85.8|10.3|3.8|2.9|17.1|93.6|
|decode_test_decode_tsd_lm_word7184|760|32771|86.5|9.7|3.8|2.6|16.1|91.7|
|decode_train_dev_decode_alsd|100|4007|86.1|11.4|2.5|2.9|16.7|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.5|11.1|2.4|2.6|16.0|96.0|
|decode_train_dev_decode_default|100|4007|85.7|11.8|2.5|2.9|17.2|94.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|86.9|11.0|2.2|2.6|15.7|93.0|
|decode_train_dev_decode_nsc|100|4007|85.8|11.7|2.5|2.8|17.0|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.2|11.2|2.5|2.6|16.3|95.0|
|decode_train_dev_decode_tsd|100|4007|85.7|11.6|2.7|2.5|16.8|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.1|11.1|2.8|2.0|15.9|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|64.6|34.5|1.0|0.5|36.0|93.4|
|decode_test_decode_alsd_lm_word7184|760|7722|67.3|31.6|1.1|0.4|33.2|91.7|
|decode_test_decode_default|760|7722|64.7|34.5|0.8|0.6|35.8|93.3|
|decode_test_decode_default_lm_word7184|760|7722|67.3|31.8|0.8|0.5|33.1|91.7|
|decode_test_decode_nsc|760|7722|64.4|34.7|0.9|0.5|36.2|93.6|
|decode_test_decode_nsc_lm_word7184|760|7722|67.2|31.9|0.9|0.5|33.3|92.2|
|decode_test_decode_tsd|760|7722|64.4|34.7|0.9|0.5|36.1|93.6|
|decode_test_decode_tsd_lm_word7184|760|7722|67.1|31.9|1.0|0.4|33.3|91.7|
|decode_train_dev_decode_alsd|100|927|62.9|37.0|0.1|0.4|37.5|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|65.5|34.5|0.0|0.2|34.7|96.0|
|decode_train_dev_decode_default|100|927|63.3|36.7|0.0|0.4|37.1|94.0|
|decode_train_dev_decode_default_lm_word7184|100|927|66.6|33.4|0.0|0.3|33.8|93.0|
|decode_train_dev_decode_nsc|100|927|62.8|37.2|0.0|0.3|37.5|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|65.2|34.8|0.0|0.2|35.1|95.0|
|decode_train_dev_decode_tsd|100|927|62.7|37.3|0.0|0.3|37.6|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|65.5|34.5|0.0|0.1|34.6|96.0|

# Transformer-Transducer (enc: 3 x TDNN-TDNN-Transformer, dec: 1 x CausalConv1d-Transformer)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to tdnn_transformer_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1Fds3PYJtSCXmvZuNDNaWCud-FWquFomH
  - training config file: `conf/tuning/transducer/train_custom_tt.yaml`
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
|decode_test_decode_alsd|760|32771|87.4|9.4|3.1|5.1|17.7|100.0|
|decode_test_decode_alsd_lm_word7184|760|32771|88.0|9.0|3.1|5.0|17.0|100.0|
|decode_test_decode_default|760|32771|87.2|9.5|3.3|4.9|17.7|100.0|
|decode_test_decode_default_lm_word7184|760|32771|88.1|8.7|3.2|4.8|16.7|100.0|
|decode_test_decode_nsc|760|32771|87.3|9.4|3.3|5.0|17.7|100.0|
|decode_test_decode_nsc_lm_word7184|760|32771|88.0|8.9|3.1|5.0|17.0|100.0|
|decode_test_decode_tsd|760|32771|87.4|9.2|3.4|4.8|17.5|100.0|
|decode_test_decode_tsd_lm_word7184|760|32771|88.1|8.5|3.4|4.6|16.5|100.0|
|decode_train_dev_decode_alsd|100|4007|86.8|11.3|2.0|5.5|18.7|100.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|87.1|10.9|2.0|5.1|18.1|100.0|
|decode_train_dev_decode_default|100|4007|86.7|11.2|2.1|5.1|18.4|100.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|87.3|10.6|2.2|5.1|17.8|100.0|
|decode_train_dev_decode_nsc|100|4007|87.2|10.8|2.0|5.1|17.9|100.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|87.4|10.6|2.0|5.3|17.9|100.0|
|decode_train_dev_decode_tsd|100|4007|86.8|11.1|2.1|5.1|18.3|100.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|87.1|10.6|2.3|4.8|17.7|100.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|67.5|31.6|0.9|0.6|33.1|91.2|
|decode_test_decode_alsd_lm_word7184|760|7722|69.7|29.4|0.9|0.5|30.8|88.3|
|decode_test_decode_default|760|7722|67.0|32.1|0.9|0.6|33.5|90.3|
|decode_test_decode_default_lm_word7184|760|7722|70.1|28.9|1.0|0.5|30.4|87.0|
|decode_test_decode_nsc|760|7722|67.1|32.0|0.9|0.6|33.5|90.4|
|decode_test_decode_nsc_lm_word7184|760|7722|69.4|29.8|0.9|0.5|31.2|88.8|
|decode_test_decode_tsd|760|7722|67.8|31.3|0.9|0.6|32.8|90.0|
|decode_test_decode_tsd_lm_word7184|760|7722|70.5|28.5|1.0|0.5|30.0|86.4|
|decode_train_dev_decode_alsd|100|927|64.2|35.8|0.0|0.5|36.4|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|67.0|32.9|0.1|0.4|33.4|94.0|
|decode_train_dev_decode_default|100|927|64.4|35.5|0.1|0.5|36.1|97.0|
|decode_train_dev_decode_default_lm_word7184|100|927|67.3|32.6|0.1|0.4|33.1|95.0|
|decode_train_dev_decode_nsc|100|927|65.2|34.8|0.0|0.4|35.3|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|67.3|32.6|0.1|0.4|33.1|93.0|
|decode_train_dev_decode_tsd|100|927|64.3|35.6|0.1|0.5|36.2|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|67.2|32.7|0.1|0.2|33.0|93.0|

# Conformer-Transducer (enc: Conv2DSubsampling + 8 x Conformer, dec: 2 x Transformer)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to conformer_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1AoQyTlFixQvf3KO3Da8EiSW33-KjH4if
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
|decode_test_decode_alsd|760|32771|88.2|9.0|2.8|2.5|14.3|91.2|
|decode_test_decode_alsd_lm_word7184|760|32771|88.8|8.4|2.8|2.4|13.6|88.9|
|decode_test_decode_default|760|32771|88.2|9.1|2.7|2.5|14.3|91.4|
|decode_test_decode_default_lm_word7184|760|32771|88.8|8.5|2.7|2.4|13.5|88.9|
|decode_test_decode_nsc|760|32771|88.2|9.0|2.8|2.5|14.3|92.1|
|decode_test_decode_nsc_lm_word7184|760|32771|88.7|8.5|2.8|2.4|13.7|90.3|
|decode_test_decode_tsd|760|32771|88.1|9.0|2.9|2.5|14.4|92.0|
|decode_test_decode_tsd_lm_word7184|760|32771|88.7|8.3|3.0|2.2|13.5|89.3|
|decode_train_dev_decode_alsd|100|4007|89.0|9.4|1.7|2.0|13.0|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|89.6|8.8|1.6|1.6|12.1|96.0|
|decode_train_dev_decode_default|100|4007|89.0|9.3|1.6|1.8|12.8|97.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.6|8.8|1.5|1.7|12.1|95.0|
|decode_train_dev_decode_nsc|100|4007|88.8|9.4|1.7|2.0|13.2|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|89.3|9.0|1.7|1.6|12.3|97.0|
|decode_train_dev_decode_tsd|100|4007|88.9|9.4|1.7|2.0|13.1|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|89.4|8.8|1.8|1.5|12.2|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|67.2|32.1|0.7|0.6|33.4|91.2|
|decode_test_decode_alsd_lm_word7184|760|7722|69.8|29.4|0.8|0.5|30.7|88.9|
|decode_test_decode_default|760|7722|67.2|32.2|0.6|0.5|33.3|91.4|
|decode_test_decode_default_lm_word7184|760|7722|69.7|29.7|0.6|0.4|30.7|88.9|
|decode_test_decode_nsc|760|7722|67.1|32.3|0.6|0.5|33.4|92.1|
|decode_test_decode_nsc_lm_word7184|760|7722|69.3|30.0|0.7|0.5|31.1|90.3|
|decode_test_decode_tsd|760|7722|66.8|32.5|0.6|0.5|33.6|92.0|
|decode_test_decode_tsd_lm_word7184|760|7722|69.7|29.4|0.8|0.4|30.7|89.3|
|decode_train_dev_decode_alsd|100|927|67.4|32.6|0.0|0.0|32.6|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|69.8|30.2|0.0|0.0|30.2|96.0|
|decode_train_dev_decode_default|100|927|67.7|32.3|0.0|0.0|32.3|97.0|
|decode_train_dev_decode_default_lm_word7184|100|927|69.7|30.3|0.0|0.0|30.3|95.0|
|decode_train_dev_decode_nsc|100|927|66.9|33.1|0.0|0.0|33.1|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|69.5|30.4|0.1|0.0|30.5|97.0|
|decode_train_dev_decode_tsd|100|927|67.2|32.8|0.0|0.0|32.8|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|69.5|30.3|0.2|0.0|30.5|96.0|

# Conformer/RNN-Transducer (enc: Conv2DSubsampling + 8 x Conformer, dec: 1 x LSTM)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to conformer-rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1pIVZAd3P1eyYR-yH-mCg64hpUoRm8lxG
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
|decode_test_decode_alsd|760|32771|88.6|8.9|2.6|2.5|14.0|90.3|
|decode_test_decode_alsd_lm_word7184|760|32771|89.1|8.4|2.5|2.4|13.3|88.4|
|decode_test_decode_default|760|32771|88.5|8.9|2.6|2.5|14.0|91.3|
|decode_test_decode_default_lm_word7184|760|32771|89.2|8.3|2.5|2.4|13.2|88.9|
|decode_test_decode_nsc|760|32771|88.4|8.9|2.7|2.6|14.2|91.4|
|decode_test_decode_nsc_lm_word7184|760|32771|89.1|8.4|2.5|2.5|13.4|88.4|
|decode_test_decode_tsd|760|32771|88.4|8.9|2.8|2.5|14.2|92.1|
|decode_test_decode_tsd_lm_word7184|760|32771|89.0|8.3|2.7|2.3|13.3|89.6|
|decode_train_dev_decode_alsd|100|4007|88.2|10.0|1.8|2.1|14.0|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|88.5|9.7|1.8|1.9|13.4|93.0|
|decode_train_dev_decode_default|100|4007|88.2|10.1|1.7|1.9|13.7|97.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|88.5|9.7|1.8|1.9|13.4|92.0|
|decode_train_dev_decode_nsc|100|4007|88.1|10.1|1.8|2.1|14.0|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|88.5|9.6|1.9|2.0|13.5|96.0|
|decode_train_dev_decode_tsd|100|4007|88.2|10.0|1.8|2.0|13.9|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|88.2|9.8|2.0|1.8|13.6|93.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|69.1|30.4|0.6|0.5|31.4|90.3|
|decode_test_decode_alsd_lm_word7184|760|7722|71.2|28.2|0.6|0.5|29.2|88.4|
|decode_test_decode_default|760|7722|69.0|30.5|0.5|0.5|31.5|91.3|
|decode_test_decode_default_lm_word7184|760|7722|71.3|28.2|0.5|0.5|29.2|88.9|
|decode_test_decode_nsc|760|7722|68.7|30.8|0.5|0.5|31.8|91.4|
|decode_test_decode_nsc_lm_word7184|760|7722|70.8|28.6|0.5|0.5|29.7|88.4|
|decode_test_decode_tsd|760|7722|68.7|30.7|0.6|0.6|31.8|92.1|
|decode_test_decode_tsd_lm_word7184|760|7722|71.2|28.2|0.6|0.5|29.3|89.6|
|decode_train_dev_decode_alsd|100|927|67.2|32.8|0.0|0.0|32.8|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|68.8|31.2|0.0|0.1|31.3|93.0|
|decode_train_dev_decode_default|100|927|67.4|32.6|0.0|0.0|32.6|97.0|
|decode_train_dev_decode_default_lm_word7184|100|927|68.7|31.3|0.0|0.0|31.3|92.0|
|decode_train_dev_decode_nsc|100|927|67.2|32.8|0.0|0.1|32.9|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|68.9|31.1|0.0|0.1|31.2|96.0|
|decode_train_dev_decode_tsd|100|927|67.3|32.7|0.0|0.1|32.8|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|68.9|31.0|0.1|0.0|31.1|93.0|

# Conformer/RNN-Transducer (enc: 3 x TDNN-TDNN-Conformer, dec: 1 x LSTM)

- Environments
  - date: `Fri Oct 16 20:22:19 CEST 2020`
  - python version: `3.7.6 (default, Jan  8 2020, 19:59:22)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.3`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.4.0`
  - Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- Model files (archived to rnn_transducer.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/open?id=1xFizgC6k9r4-UkcAk7TRFB7bkUt4L2Bd
  - training config file: `conf/tuning/transducer/train_custom_ct.yaml`
  - decoding config file: `conf/tuning/transducer/decode_default.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_custom_ct/results/model.loss.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_custom_ct/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm_word7184/model.json`
  - dict file: `data/lang_1char/`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|89.5|8.2|2.4|2.7|13.2|87.8|
|decode_test_decode_alsd_lm_word7184|760|32771|90.1|7.7|2.2|2.5|12.5|85.4|
|decode_test_decode_default|760|32771|89.4|8.3|2.3|2.5|13.1|87.8|
|decode_test_decode_default_lm_word7184|760|32771|90.1|7.7|2.2|2.4|12.3|85.3|
|decode_test_decode_nsc|760|32771|89.4|8.3|2.3|2.7|13.3|89.5|
|decode_test_decode_nsc_lm_word7184|760|32771|89.9|7.8|2.3|2.6|12.7|87.2|
|decode_test_decode_tsd|760|32771|89.4|8.2|2.4|2.6|13.2|88.7|
|decode_test_decode_tsd_lm_word7184|760|32771|90.1|7.6|2.3|2.3|12.3|85.7|
|decode_train_dev_decode_alsd|100|4007|90.5|7.8|1.7|1.9|11.3|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|90.9|7.4|1.7|1.9|11.0|93.0|
|decode_train_dev_decode_default|100|4007|90.2|7.9|1.8|1.9|11.6|93.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|90.9|7.4|1.6|1.8|10.9|92.0|
|decode_train_dev_decode_nsc|100|4007|90.2|8.0|1.8|2.0|11.8|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|90.3|7.9|1.7|2.0|11.6|92.0|
|decode_train_dev_decode_tsd|100|4007|90.1|7.9|1.9|1.9|11.7|94.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|90.7|7.3|1.9|1.7|11.0|92.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|71.5|27.8|0.7|0.5|29.0|87.8|
|decode_test_decode_alsd_lm_word7184|760|7722|73.7|25.7|0.7|0.5|26.8|85.4|
|decode_test_decode_default|760|7722|71.5|27.9|0.6|0.5|29.0|87.8|
|decode_test_decode_default_lm_word7184|760|7722|73.8|25.6|0.6|0.4|26.6|85.3|
|decode_test_decode_nsc|760|7722|71.1|28.4|0.6|0.6|29.5|89.3|
|decode_test_decode_nsc_lm_word7184|760|7722|72.9|26.5|0.6|0.5|27.7|87.1|
|decode_test_decode_tsd|760|7722|71.5|27.9|0.6|0.5|29.0|88.0|
|decode_test_decode_tsd_lm_word7184|760|7722|74.1|25.3|0.7|0.4|26.4|85.3|
|decode_train_dev_decode_alsd|100|927|71.6|28.4|0.0|0.0|28.4|93.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|73.6|26.4|0.0|0.0|26.4|93.0|
|decode_train_dev_decode_default|100|927|71.3|28.7|0.0|0.0|28.7|93.0|
|decode_train_dev_decode_default_lm_word7184|100|927|73.6|26.4|0.0|0.0|26.4|92.0|
|decode_train_dev_decode_nsc|100|927|70.7|29.3|0.0|0.2|29.6|94.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|71.5|28.5|0.0|0.2|28.7|92.0|
|decode_train_dev_decode_tsd|100|927|70.9|29.1|0.0|0.1|29.2|94.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|74.0|25.9|0.1|0.3|26.3|92.0|

# pytorch CTC model (4 x BLSTMP)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight1.0|760|32771|80.1|13.0|6.8|2.3|22.2|98.7|
|decode_test_decode_ctcweight1.0_lm|760|32771|84.2|12.0|3.8|3.3|19.1|91.8|
|decode_test_decode_ctcweight1.0_lm_word7184|760|32771|83.0|12.7|4.3|3.2|20.2|93.9|
|decode_train_dev_decode_ctcweight1.0|100|4007|82.6|12.0|5.4|1.7|19.1|99.0|
|decode_train_dev_decode_ctcweight1.0_lm|100|4007|85.3|11.5|3.2|2.1|16.9|93.0|
|decode_train_dev_decode_ctcweight1.0_lm_word7184|100|4007|84.1|12.3|3.5|2.2|18.1|99.0|
