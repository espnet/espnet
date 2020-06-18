# pytorch Transducer (enc: VGG2L + 6 x Transformer, dec: 1 x LSTM)

- Environments
  - date: `Fri Apr  3 10:34:40 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `d19105b0793b1a9c6e11293f2f95f85699632367`
  - Commit date: `Fri Apr 3 10:09:03 2020 +0200`

- Model files (archived to model.tar.gz by `$ pack_model.sh`)
  - model link: https://drive.google.com/file/d/1mW2ANumwGozam7mt2_0JSpUAWIFfvyac
  - training config file: `conf/tuning/transducer/train_transformer-rnn_transducer.yaml`
  - decoding config file: `conf/tuning/transducer/decode_transducer.yaml`
  - cmvn file: `data/train_nodev/cmvn.ark`
  - e2e file: `exp/train_nodev_pytorch_train_transformer-rnn_transducer/results/model.last5.avg.best`
  - e2e JSON file: `exp/train_nodev_pytorch_train_transformer-rnn_transducer/results/model.json`
  - lm file: `exp/train_rnnlm_pytorch_lm_word7184/rnnlm.model.best`
  - lm JSON file: `exp/train_rnnlm_pytorch_lm/model.json`
  - dict file: `data/lang_1char`

- Results (obtained with `$ show_results.sh`)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|32771|85.4|10.8|3.8|3.0|17.6|94.2|
|decode_test_decode_transducer_lm|760|32771|85.6|10.6|3.8|3.0|17.4|93.3|
|decode_test_decode_transducer_lm_word7184|760|32771|86.1|10.2|3.7|2.8|16.7|92.1|
|decode_train_dev_decode_transducer|100|4007|84.7|12.8|2.6|3.2|18.6|97.0|
|decode_train_dev_decode_transducer_lm|100|4007|84.7|12.6|2.7|3.3|18.6|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|84.9|12.4|2.7|3.3|18.4|97.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|63.7|35.4|0.9|0.5|36.8|94.2|
|decode_test_decode_transducer_lm|760|7722|64.8|34.0|1.2|0.5|35.7|93.3|
|decode_test_decode_transducer_lm_word7184|760|7722|66.5|32.3|1.2|0.4|33.9|92.1|
|decode_train_dev_decode_transducer|100|927|59.8|39.9|0.3|0.3|40.6|97.0|
|decode_train_dev_decode_transducer_lm|100|927|60.7|38.6|0.6|0.3|39.6|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|62.2|37.2|0.5|0.2|38.0|97.0|

# pytorch Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

- Environments
  - date: `Fri Apr  3 10:34:40 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `d19105b0793b1a9c6e11293f2f95f85699632367`
  - Commit date: `Fri Apr 3 10:09:03 2020 +0200`

- Results (obtained with `$ show_results.sh`)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|32771|84.0|11.8|4.2|3.2|19.2|96.7|
|decode_test_decode_transducer_lm|760|32771|84.8|11.0|4.2|3.0|18.2|95.9|
|decode_test_decode_transducer_lm_word7184|760|32771|85.3|10.4|4.3|2.9|17.6|94.3|
|decode_train_dev_decode_transducer|100|4007|84.5|12.6|2.9|3.1|18.6|100.0|
|decode_train_dev_decode_transducer_lm|100|4007|85.7|11.6|2.8|2.4|16.8|98.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|86.2|11.2|2.6|2.2|16.0|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|55.9|43.1|0.9|0.5|44.6|96.7|
|decode_test_decode_transducer_lm|760|7722|59.4|39.3|1.3|0.4|41.1|95.9|
|decode_test_decode_transducer_lm_word7184|760|7722|62.0|36.6|1.4|0.5|38.5|94.3|
|decode_train_dev_decode_transducer|100|927|57.0|42.8|0.2|0.1|43.1|100.0|
|decode_train_dev_decode_transducer_lm|100|927|62.7|36.8|0.5|0.1|37.4|98.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|64.6|34.8|0.5|0.1|35.5|96.0|

# pytorch Transducer (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Fri Apr  3 10:34:40 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `d19105b0793b1a9c6e11293f2f95f85699632367`
  - Commit date: `Fri Apr 3 10:09:03 2020 +0200`

- Results (obtained with `$ show_results.sh`)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|32771|84.9|11.5|3.6|3.3|18.4|93.2|
|decode_test_decode_transducer_lm|760|32771|84.9|10.8|4.3|3.0|18.1|92.5|
|decode_test_decode_transducer_lm_word7184|760|32771|85.3|10.7|4.1|2.9|17.7|91.4|
|decode_train_dev_decode_transducer|100|4007|85.1|11.7|3.2|2.5|17.3|97.0|
|decode_train_dev_decode_transducer_lm|100|4007|85.3|11.5|3.2|2.2|16.9|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|85.4|11.0|3.6|2.2|16.8|95.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|61.8|37.5|0.7|0.6|38.8|93.2|
|decode_test_decode_transducer_lm|760|7722|63.5|35.1|1.4|0.5|37.0|92.5|
|decode_test_decode_transducer_lm_word7184|760|7722|64.8|33.9|1.3|0.5|35.7|91.4|
|decode_train_dev_decode_transducer|100|927|60.8|38.8|0.3|0.1|39.3|97.0|
|decode_train_dev_decode_transducer_lm|100|927|62.9|36.7|0.4|0.1|37.2|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|64.2|35.2|0.6|0.1|35.9|95.0|

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
