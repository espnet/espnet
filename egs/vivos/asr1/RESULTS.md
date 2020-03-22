# pytorch Transducer (enc: VGG2L + 6 x Transformer, dec: 1 x LSTM)

- Environments:
  - date: `Sun Feb  9 20:50:30 CET 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `b19ae06fa0ee36d9a65ca701dd169d52cf1a24fd`
  - Commit date: `Fri Feb 7 16:32:23 2020 +0100

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
|decode_test_decode_transducer|760|32771|85.3|11.0|3.7|3.1|17.8|94.3|
|decode_test_decode_transducer_lm|760|32771|85.9|10.5|3.6|3.0|17.0|92.4|
|decode_test_decode_transducer_lm_word7184|760|32771|86.2|10.2|3.6|2.9|16.7|91.4|
|decode_train_dev_decode_transducer|100|4007|84.7|12.6|2.7|3.2|18.5|97.0|
|decode_train_dev_decode_transducer_lm|100|4007|85.4|12.3|2.3|2.9|17.5|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|86.0|11.5|2.4|2.9|16.9|97.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|63.4|35.7|0.9|0.5|37.1|94.3|
|decode_test_decode_transducer_lm|760|7722|65.2|33.7|1.1|0.4|35.2|92.4|
|decode_test_decode_transducer_lm_word7184|760|7722|66.7|32.3|1.0|0.4|33.7|91.4|
|decode_train_dev_decode_transducer|100|927|61.4|38.5|0.1|0.2|38.8|97.0|
|decode_train_dev_decode_transducer_lm|100|927|63.1|36.9|0.0|0.1|37.0|97.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|65.2|34.6|0.2|0.0|34.8|97.0|

# pytorch Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

- Environments
  - date: `Sun Feb  9 20:50:30 CET 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `b19ae06fa0ee36d9a65ca701dd169d52cf1a24fd`
  - Commit date: `Fri Feb 7 16:32:23 2020 +0100

- Results (obtained with `$ show_results.sh`)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|32771|84.3|11.7|4.0|3.4|19.1|97.1|
|decode_test_decode_transducer_lm|760|32771|85.0|11.0|4.0|3.1|18.1|95.9|
|decode_test_decode_transducer_lm_word7184|760|32771|85.6|10.4|4.0|2.9|17.4|94.3|
|decode_train_dev_decode_transducer|100|4007|84.6|12.8|2.6|2.9|18.3|100.0|
|decode_train_dev_decode_transducer_lm|100|4007|85.7|11.6|2.7|2.4|16.7|98.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|86.2|11.1|2.7|2.1|15.9|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|56.2|42.8|1.0|0.5|44.3|97.1|
|decode_test_decode_transducer_lm|760|7722|59.4|39.4|1.2|0.5|41.1|95.9|
|decode_test_decode_transducer_lm_word7184|760|7722|62.1|36.7|1.2|0.5|38.4|94.3|
|decode_train_dev_decode_transducer|100|927|58.1|41.6|0.2|0.1|42.0|100.0|
|decode_train_dev_decode_transducer_lm|100|927|62.7|36.9|0.4|0.1|37.4|98.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|64.5|35.0|0.5|0.1|35.6|96.0|

# pytorch Transducer (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Sun Feb  9 20:50:30 CET 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `b19ae06fa0ee36d9a65ca701dd169d52cf1a24fd`
  - Commit date: `Fri Feb 7 16:32:23 2020 +0100

- Results (obtained with `$ show_results.sh`)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|32771|84.5|11.7|3.8|3.2|18.7|94.1|
|decode_test_decode_transducer_lm|760|32771|84.9|11.2|3.8|3.0|18.1|93.3|
|decode_test_decode_transducer_lm_word7184|760|32771|85.2|11.0|3.8|3.0|17.8|92.2|
|decode_train_dev_decode_transducer|100|4007|86.1|11.1|2.9|2.1|16.0|96.0|
|decode_train_dev_decode_transducer_lm|100|4007|86.3|10.9|2.8|2.0|15.7|96.0|
|decode_train_dev_decode_transducer_lm_word7184|100|4007|86.6|10.4|2.9|1.8|15.2|96.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_transducer|760|7722|61.0|38.2|0.7|0.6|39.6|94.1|
|decode_test_decode_transducer_lm|760|7722|62.8|36.3|0.8|0.5|37.7|93.3|
|decode_test_decode_transducer_lm_word7184|760|7722|64.2|35.0|0.9|0.5|36.4|92.2|
|decode_train_dev_decode_transducer|100|927|63.6|36.1|0.2|0.0|36.4|96.0|
|decode_train_dev_decode_transducer_lm|100|927|64.9|34.8|0.2|0.0|35.1|96.0|
|decode_train_dev_decode_transducer_lm_word7184|100|927|67.0|32.8|0.2|0.0|33.0|96.0|

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