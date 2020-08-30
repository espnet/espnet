# RNN-Transducer (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Sat Aug 22 16:14:35 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `077c8970afb477e059932f7243a9b728c8ab1a69`
  - Commit date: `Fri Aug 21 16:32:27 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|85.3|11.5|3.2|3.6|18.3|93.8|
|decode_test_decode_alsd_lm_word7184|760|32771|86.2|10.7|3.1|3.5|17.3|92.0|
|decode_test_decode_default|760|32771|84.6|11.6|3.8|3.2|18.6|93.3|
|decode_test_decode_default_lm_word7184|760|32771|84.4|10.7|4.9|2.8|18.4|92.0|
|decode_test_decode_nsc|760|32771|85.2|11.4|3.4|3.5|18.3|93.0|
|decode_test_decode_nsc_lm_word7184|760|32771|85.9|10.8|3.3|3.3|17.4|92.0|
|decode_test_decode_tsd|760|32771|85.0|11.4|3.5|3.2|18.2|93.6|
|decode_test_decode_tsd_lm_word7184|760|32771|86.0|10.5|3.6|3.0|17.0|90.9|
|decode_train_dev_decode_alsd|100|4007|85.8|11.5|2.7|2.6|16.8|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.7|10.8|2.5|2.2|15.5|97.0|
|decode_train_dev_decode_default|100|4007|84.8|11.9|3.3|2.3|17.5|98.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|84.5|11.2|4.3|2.1|17.6|93.0|
|decode_train_dev_decode_nsc|100|4007|85.5|11.6|2.8|2.5|16.9|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.3|10.9|2.8|2.3|16.0|96.0|
|decode_train_dev_decode_tsd|100|4007|85.2|11.8|3.0|2.2|17.0|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.4|10.7|2.9|1.9|15.5|95.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|62.2|37.3|0.5|0.6|38.4|93.8|
|decode_test_decode_alsd_lm_word7184|760|7722|65.3|34.1|0.6|0.6|35.3|92.0|
|decode_test_decode_default|760|7722|61.5|37.7|0.8|0.5|39.0|93.3|
|decode_test_decode_default_lm_word7184|760|7722|64.0|33.9|2.0|0.4|36.4|92.0|
|decode_test_decode_nsc|760|7722|61.9|37.5|0.6|0.6|38.7|93.0|
|decode_test_decode_nsc_lm_word7184|760|7722|64.7|34.6|0.7|0.6|35.9|92.0|
|decode_test_decode_tsd|760|7722|62.1|37.3|0.6|0.6|38.5|93.6|
|decode_test_decode_tsd_lm_word7184|760|7722|65.9|33.3|0.8|0.5|34.6|90.9|
|decode_train_dev_decode_alsd|100|927|62.2|37.6|0.1|0.0|37.8|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|66.0|33.9|0.1|0.0|34.0|97.0|
|decode_train_dev_decode_default|100|927|60.3|39.4|0.3|0.1|39.8|98.0|
|decode_train_dev_decode_default_lm_word7184|100|927|63.0|35.8|1.2|0.1|37.1|93.0|
|decode_train_dev_decode_nsc|100|927|61.7|38.2|0.1|0.2|38.5|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|64.7|35.1|0.2|0.2|35.5|96.0|
|decode_train_dev_decode_tsd|100|927|61.6|38.3|0.1|0.0|38.4|97.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|65.8|34.1|0.1|0.0|34.2|95.0|

# RNN-Transducer w/ att (enc: 4 x BLSTMP, dec: 1 x LSTM)

- Environments
  - date: `Sat Aug 22 16:14:35 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `077c8970afb477e059932f7243a9b728c8ab1a69`
  - Commit date: `Fri Aug 21 16:32:27 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|84.9|11.8|3.3|3.5|18.7|93.8|
|decode_test_decode_alsd_lm_word7184|760|32771|85.4|11.3|3.3|3.3|17.9|92.5|
|decode_test_decode_default|760|32771|84.3|11.9|3.8|3.1|18.7|94.5|
|decode_test_decode_default_lm_word7184|760|32771|84.1|11.0|4.9|2.7|18.6|92.1|
|decode_test_decode_nsc|760|32771|84.7|11.8|3.5|3.4|18.7|94.2|
|decode_test_decode_nsc_lm_word7184|760|32771|85.3|11.2|3.4|3.2|17.9|93.2|
|decode_test_decode_tsd|760|32771|84.7|11.7|3.6|3.1|18.5|93.7|
|decode_test_decode_tsd_lm_word7184|760|32771|85.2|11.0|3.8|2.8|17.6|92.6|
|decode_train_dev_decode_alsd|100|4007|86.3|11.3|2.4|2.3|16.0|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|86.7|10.8|2.5|2.4|15.7|93.0|
|decode_train_dev_decode_default|100|4007|85.5|11.5|3.0|2.1|16.6|95.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.6|10.4|4.0|1.8|16.2|95.0|
|decode_train_dev_decode_nsc|100|4007|85.9|11.3|2.7|2.3|16.4|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.4|10.9|2.6|2.2|15.8|96.0|
|decode_train_dev_decode_tsd|100|4007|85.8|11.4|2.7|2.1|16.2|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.1|10.7|3.2|1.9|15.7|95.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|60.8|38.7|0.5|0.6|39.8|93.8|
|decode_test_decode_alsd_lm_word7184|760|7722|63.6|35.8|0.7|0.5|37.0|92.5|
|decode_test_decode_default|760|7722|60.3|38.9|0.8|0.5|40.2|94.5|
|decode_test_decode_default_lm_word7184|760|7722|62.7|35.3|2.0|0.4|37.7|92.1|
|decode_test_decode_nsc|760|7722|60.5|38.9|0.6|0.6|40.0|94.2|
|decode_test_decode_nsc_lm_word7184|760|7722|63.1|36.2|0.7|0.5|37.4|93.2|
|decode_test_decode_tsd|760|7722|60.9|38.5|0.6|0.5|39.6|93.7|
|decode_test_decode_tsd_lm_word7184|760|7722|63.9|35.1|1.0|0.5|36.6|92.6|
|decode_train_dev_decode_alsd|100|927|63.5|36.5|0.0|0.0|36.5|97.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|66.1|33.9|0.0|0.0|33.9|93.0|
|decode_train_dev_decode_default|100|927|61.9|38.0|0.1|0.0|38.1|95.0|
|decode_train_dev_decode_default_lm_word7184|100|927|65.3|33.8|1.0|0.0|34.7|95.0|
|decode_train_dev_decode_nsc|100|927|62.8|37.2|0.0|0.1|37.3|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|64.8|35.1|0.1|0.0|35.2|96.0|
|decode_train_dev_decode_tsd|100|927|62.8|37.2|0.0|0.1|37.3|95.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|65.9|33.9|0.2|0.0|34.1|95.0|

# Transformer-Transducer (enc: VGG2L + 6 x TDNN-Transformer, dec: 2 x CausalConv1d-Transformer)

- Environments
  - date: `Sat Aug 22 16:14:35 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `077c8970afb477e059932f7243a9b728c8ab1a69`
  - Commit date: `Fri Aug 21 16:32:27 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|86.4|9.9|3.7|4.8|18.4|100.0|
|decode_test_decode_alsd_lm_word7184|760|32771|87.0|9.2|3.8|4.6|17.6|99.9|
|decode_test_decode_default|760|32771|85.7|9.8|4.5|4.4|18.8|100.0|
|decode_test_decode_default_lm_word7184|760|32771|85.3|9.3|5.4|4.2|19.0|100.0|
|decode_test_decode_nsc|760|32771|86.0|10.0|4.1|4.7|18.8|100.0|
|decode_test_decode_nsc_lm_word7184|760|32771|86.4|9.4|4.2|4.6|18.2|100.0|
|decode_test_decode_tsd|760|32771|85.8|9.8|4.4|4.5|18.8|100.0|
|decode_test_decode_tsd_lm_word7184|760|32771|86.3|9.0|4.8|4.2|18.0|100.0|
|decode_train_dev_decode_alsd|100|4007|88.2|9.7|2.1|4.9|16.6|100.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|88.6|9.4|2.0|4.8|16.2|100.0|
|decode_train_dev_decode_default|100|4007|87.2|10.0|2.8|4.8|17.5|100.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|87.8|9.6|2.6|4.5|16.7|100.0|
|decode_train_dev_decode_nsc|100|4007|87.8|10.0|2.2|4.6|16.8|100.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|88.3|9.5|2.2|4.7|16.4|100.0|
|decode_train_dev_decode_tsd|100|4007|87.7|10.0|2.3|4.8|17.0|100.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|88.4|9.2|2.4|4.4|16.0|100.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|65.7|33.3|1.0|0.5|34.8|90.5|
|decode_test_decode_alsd_lm_word7184|760|7722|68.0|30.8|1.2|0.4|32.4|88.3|
|decode_test_decode_default|760|7722|65.2|33.3|1.6|0.4|35.3|91.1|
|decode_test_decode_default_lm_word7184|760|7722|66.6|30.7|2.6|0.3|33.7|88.6|
|decode_test_decode_nsc|760|7722|65.0|33.7|1.2|0.5|35.5|91.7|
|decode_test_decode_nsc_lm_word7184|760|7722|66.9|31.6|1.5|0.5|33.6|90.1|
|decode_test_decode_tsd|760|7722|65.4|33.1|1.5|0.5|35.1|91.4|
|decode_test_decode_tsd_lm_word7184|760|7722|67.9|30.1|2.0|0.4|32.4|88.4|
|decode_train_dev_decode_alsd|100|927|68.0|32.0|0.0|0.3|32.4|92.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|69.9|30.0|0.1|0.4|30.5|91.0|
|decode_train_dev_decode_default|100|927|66.7|33.0|0.3|0.4|33.8|94.0|
|decode_train_dev_decode_default_lm_word7184|100|927|69.4|30.2|0.4|0.5|31.2|91.0|
|decode_train_dev_decode_nsc|100|927|67.4|32.6|0.0|0.3|32.9|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|69.3|30.6|0.1|0.4|31.2|93.0|
|decode_train_dev_decode_tsd|100|927|67.5|32.5|0.0|0.3|32.8|94.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|70.1|29.7|0.2|0.3|30.2|91.0|

# Transformer-Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

- Environments
  - date: `Sat Aug 22 16:14:35 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `077c8970afb477e059932f7243a9b728c8ab1a69`
  - Commit date: `Fri Aug 21 16:32:27 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|84.9|10.7|4.4|3.2|18.3|95.5|
|decode_test_decode_alsd_lm_word7184|760|32771|85.8|9.6|4.5|2.8|16.9|93.8|
|decode_test_decode_default|760|32771|84.6|10.8|4.6|3.3|18.6|96.1|
|decode_test_decode_default_lm_word7184|760|32771|85.0|9.7|5.4|2.6|17.6|94.6|
|decode_test_decode_nsc|760|32771|84.6|10.8|4.6|3.3|18.6|95.9|
|decode_test_decode_nsc_lm_word7184|760|32771|85.5|9.8|4.7|2.8|17.3|94.1|
|decode_test_decode_tsd|760|32771|84.3|10.8|4.9|3.0|18.8|96.2|
|decode_test_decode_tsd_lm_word7184|760|32771|84.9|9.5|5.6|2.5|17.6|94.6|
|decode_train_dev_decode_alsd|100|4007|85.7|11.6|2.7|2.6|16.9|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|85.9|10.9|3.2|2.3|16.4|97.0|
|decode_train_dev_decode_default|100|4007|85.1|11.5|3.5|2.8|17.8|98.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.4|10.8|3.9|2.4|17.0|98.0|
|decode_train_dev_decode_nsc|100|4007|85.2|11.8|3.0|2.7|17.5|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.0|10.8|3.2|2.2|16.3|97.0|
|decode_train_dev_decode_tsd|100|4007|84.6|11.7|3.6|2.6|18.0|98.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|85.2|10.6|4.2|2.1|16.9|98.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|58.8|39.5|1.7|0.5|41.7|95.5|
|decode_test_decode_alsd_lm_word7184|760|7722|63.2|34.7|2.0|0.4|37.2|93.8|
|decode_test_decode_default|760|7722|57.9|40.3|1.8|0.5|42.6|96.1|
|decode_test_decode_default_lm_word7184|760|7722|62.5|34.6|2.8|0.5|37.9|94.6|
|decode_test_decode_nsc|760|7722|58.0|40.2|1.8|0.5|42.5|95.9|
|decode_test_decode_nsc_lm_word7184|760|7722|62.7|35.2|2.0|0.5|37.7|94.1|
|decode_test_decode_tsd|760|7722|57.8|40.3|1.9|0.5|42.7|96.2|
|decode_test_decode_tsd_lm_word7184|760|7722|62.7|34.5|2.8|0.4|37.7|94.6|
|decode_train_dev_decode_alsd|100|927|60.4|39.2|0.4|0.0|39.6|98.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|63.6|35.4|1.0|0.0|36.4|97.0|
|decode_train_dev_decode_default|100|927|59.3|39.7|1.0|0.0|40.7|98.0|
|decode_train_dev_decode_default_lm_word7184|100|927|63.2|35.3|1.5|0.0|36.8|98.0|
|decode_train_dev_decode_nsc|100|927|59.4|39.9|0.6|0.0|40.6|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|63.4|35.5|1.1|0.0|36.6|97.0|
|decode_train_dev_decode_tsd|100|927|59.2|39.7|1.1|0.0|40.8|98.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|62.9|35.4|1.7|0.0|37.1|98.0|

# mixed RNN/Transformer-Transducer (enc: VGG2L + 6 x Transformer, dec: 1 x LSTM)

- Environments
  - date: `Sat Aug 22 16:14:35 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.6.2`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `077c8970afb477e059932f7243a9b728c8ab1a69`
  - Commit date: `Fri Aug 21 16:32:27 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|85.4|10.8|3.9|3.1|17.7|94.2|
|decode_test_decode_alsd_lm_word7184|760|32771|86.0|10.2|3.8|2.9|16.9|92.9|
|decode_test_decode_default|760|32771|85.5|10.7|3.8|2.9|17.4|94.7|
|decode_test_decode_default_lm_word7184|760|32771|85.8|10.1|4.1|2.7|16.9|93.7|
|decode_test_decode_nsc|760|32771|85.6|10.7|3.7|3.0|17.4|95.0|
|decode_test_decode_nsc_lm_word7184|760|32771|86.4|10.1|3.5|2.8|16.4|93.2|
|decode_test_decode_tsd|760|32771|85.4|10.7|3.9|2.8|17.4|94.9|
|decode_test_decode_tsd_lm_word7184|760|32771|86.1|10.0|4.0|2.5|16.5|92.9|
|decode_train_dev_decode_alsd|100|4007|86.6|11.2|2.2|2.9|16.3|92.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|87.6|10.2|2.2|2.7|15.1|91.0|
|decode_train_dev_decode_default|100|4007|86.3|11.0|2.6|2.7|16.3|96.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|86.2|10.8|3.0|2.3|16.1|91.0|
|decode_train_dev_decode_nsc|100|4007|86.0|11.4|2.5|2.8|16.8|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|87.0|10.7|2.3|2.5|15.5|92.0|
|decode_train_dev_decode_tsd|100|4007|86.0|11.2|2.8|2.7|16.7|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|86.7|10.5|2.8|2.3|15.6|93.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|63.3|35.7|1.0|0.5|37.2|94.2|
|decode_test_decode_alsd_lm_word7184|760|7722|65.7|33.2|1.1|0.5|34.8|92.9|
|decode_test_decode_default|760|7722|63.6|35.5|0.9|0.5|36.9|94.7|
|decode_test_decode_default_lm_word7184|760|7722|66.0|32.7|1.3|0.5|34.5|93.7|
|decode_test_decode_nsc|760|7722|63.6|35.5|0.8|0.6|37.0|95.0|
|decode_test_decode_nsc_lm_word7184|760|7722|66.5|32.6|0.9|0.5|34.1|93.2|
|decode_test_decode_tsd|760|7722|63.4|35.7|0.9|0.5|37.1|94.9|
|decode_test_decode_tsd_lm_word7184|760|7722|66.5|32.4|1.1|0.5|33.9|92.9|
|decode_train_dev_decode_alsd|100|927|64.1|35.8|0.1|0.4|36.4|92.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|66.8|33.0|0.2|0.3|33.5|91.0|
|decode_train_dev_decode_default|100|927|64.2|35.6|0.2|0.4|36.2|96.0|
|decode_train_dev_decode_default_lm_word7184|100|927|65.7|33.5|0.8|0.3|34.6|91.0|
|decode_train_dev_decode_nsc|100|927|63.5|36.4|0.1|0.4|36.9|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|66.1|33.7|0.2|0.3|34.2|92.0|
|decode_train_dev_decode_tsd|100|927|63.4|36.5|0.1|0.4|37.0|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|65.9|33.8|0.3|0.3|34.4|93.0|

# Conformer-Transducer (enc: Conv2DSubsampling + 8 x Conformer, dec: 2 x Conformer)

- Environments
  - date: `Wed Aug 26 11:15:29 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `46991fbc8010e66365235b1a179841e06d34a4db`
  - Commit date: `Tue Aug 25 10:58:45 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|88.1|9.1|2.8|2.7|14.7|91.7|
|decode_test_decode_alsd_lm_word7184|760|32771|88.6|8.6|2.8|2.4|13.8|88.7|
|decode_test_decode_default|760|32771|88.0|9.2|2.8|2.7|14.7|91.6|
|decode_test_decode_default_lm_word7184|760|32771|88.3|8.6|3.0|2.4|14.1|89.6|
|decode_test_decode_nsc|760|32771|87.9|9.2|2.8|2.8|14.8|91.4|
|decode_test_decode_nsc_lm_word7184|760|32771|88.4|8.8|2.9|2.5|14.2|89.9|
|decode_test_decode_tsd|760|32771|88.0|9.1|2.9|2.7|14.7|91.7|
|decode_test_decode_tsd_lm_word7184|760|32771|88.4|8.6|3.1|2.4|14.0|90.0|
|decode_train_dev_decode_alsd|100|4007|89.5|9.0|1.5|1.9|12.5|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|90.1|8.3|1.6|1.5|11.4|94.0|
|decode_train_dev_decode_default|100|4007|89.5|8.9|1.6|1.9|12.4|96.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.8|8.5|1.7|1.6|11.8|94.0|
|decode_train_dev_decode_nsc|100|4007|89.3|9.0|1.6|1.8|12.5|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|89.6|8.7|1.7|1.5|12.0|95.0|
|decode_train_dev_decode_tsd|100|4007|89.1|9.0|1.8|1.9|12.8|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|89.8|8.4|1.8|1.5|11.7|94.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|66.4|32.8|0.8|0.4|34.0|91.7|
|decode_test_decode_alsd_lm_word7184|760|7722|69.2|29.9|0.9|0.4|31.2|88.7|
|decode_test_decode_default|760|7722|66.1|33.1|0.8|0.4|34.3|91.6|
|decode_test_decode_default_lm_word7184|760|7722|68.7|30.2|1.1|0.4|31.7|89.6|
|decode_test_decode_nsc|760|7722|65.8|33.5|0.7|0.4|34.7|91.4|
|decode_test_decode_nsc_lm_word7184|760|7722|68.2|30.9|0.9|0.4|32.2|89.9|
|decode_test_decode_tsd|760|7722|66.1|33.1|0.7|0.4|34.3|91.7|
|decode_test_decode_tsd_lm_word7184|760|7722|69.0|30.0|1.0|0.4|31.4|90.0|
|decode_train_dev_decode_alsd|100|927|68.1|31.8|0.1|0.0|31.9|96.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|71.4|28.4|0.2|0.0|28.6|94.0|
|decode_train_dev_decode_default|100|927|68.0|31.9|0.1|0.0|32.0|96.0|
|decode_train_dev_decode_default_lm_word7184|100|927|70.7|29.1|0.2|0.0|29.3|94.0|
|decode_train_dev_decode_nsc|100|927|67.9|32.0|0.1|0.0|32.1|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|69.9|29.9|0.2|0.0|30.1|95.0|
|decode_train_dev_decode_tsd|100|927|67.4|32.4|0.2|0.0|32.6|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|71.1|28.7|0.2|0.0|28.9|94.0|

# mixed RNN/Conformer-Transducer (enc: Conv2DSubsampling + 8 x Conformer, dec: 1 x LSTM)

- Environments
  - date: `Wed Aug 26 11:15:29 CEST 2020`
  - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
  - espnet version: `espnet 0.9.1`
  - chainer version: `chainer 6.0.0`
  - pytorch version: `pytorch 1.0.1.post2`
  - Git hash: `46991fbc8010e66365235b1a179841e06d34a4db`
  - Commit date: `Tue Aug 25 10:58:45 2020 +0200`

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|32771|88.5|8.8|2.7|2.6|14.1|90.1|
|decode_test_decode_alsd_lm_word7184|760|32771|89.0|8.4|2.6|2.5|13.5|88.8|
|decode_test_decode_default|760|32771|88.4|8.8|2.7|2.6|14.1|90.0|
|decode_test_decode_default_lm_word7184|760|32771|88.8|8.4|2.8|2.5|13.6|88.6|
|decode_test_decode_nsc|760|32771|88.4|8.8|2.7|2.7|14.2|91.2|
|decode_test_decode_nsc_lm_word7184|760|32771|88.9|8.4|2.7|2.5|13.6|89.6|
|decode_test_decode_tsd|760|32771|88.4|8.8|2.8|2.6|14.2|91.1|
|decode_test_decode_tsd_lm_word7184|760|32771|88.8|8.3|2.9|2.4|13.5|88.6|
|decode_train_dev_decode_alsd|100|4007|87.8|10.6|1.6|2.5|14.7|95.0|
|decode_train_dev_decode_alsd_lm_word7184|100|4007|88.4|10.1|1.5|2.1|13.7|95.0|
|decode_train_dev_decode_default|100|4007|87.6|10.8|1.6|2.4|14.8|95.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|87.9|10.4|1.6|2.1|14.2|93.0|
|decode_train_dev_decode_nsc|100|4007|87.9|10.5|1.6|2.5|14.6|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|88.3|10.2|1.5|2.3|14.0|94.0|
|decode_train_dev_decode_tsd|100|4007|87.8|10.6|1.6|2.5|14.7|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|4007|88.3|9.9|1.8|1.9|13.7|94.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_alsd|760|7722|69.2|30.2|0.6|0.5|31.3|90.1|
|decode_test_decode_alsd_lm_word7184|760|7722|71.1|28.3|0.6|0.4|29.3|88.8|
|decode_test_decode_default|760|7722|68.6|30.8|0.6|0.4|31.8|90.0|
|decode_test_decode_default_lm_word7184|760|7722|70.6|28.7|0.7|0.4|29.8|88.6|
|decode_test_decode_nsc|760|7722|68.5|30.9|0.6|0.5|31.9|91.1|
|decode_test_decode_nsc_lm_word7184|760|7722|70.7|28.7|0.6|0.4|29.7|89.5|
|decode_test_decode_tsd|760|7722|68.7|30.7|0.6|0.5|31.8|90.9|
|decode_test_decode_tsd_lm_word7184|760|7722|71.0|28.4|0.7|0.5|29.5|88.4|
|decode_train_dev_decode_alsd|100|927|65.9|34.1|0.0|0.0|34.1|95.0|
|decode_train_dev_decode_alsd_lm_word7184|100|927|68.9|31.1|0.0|0.0|31.1|95.0|
|decode_train_dev_decode_default|100|927|65.4|34.6|0.0|0.0|34.6|95.0|
|decode_train_dev_decode_default_lm_word7184|100|927|67.9|32.1|0.0|0.0|32.1|93.0|
|decode_train_dev_decode_nsc|100|927|65.9|34.1|0.0|0.0|34.1|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|68.1|31.9|0.0|0.0|31.9|94.0|
|decode_train_dev_decode_tsd|100|927|65.6|34.4|0.0|0.0|34.4|96.0|
|decode_train_dev_decode_tsd_lm_word7184|100|927|68.7|31.3|0.0|0.0|31.3|94.0|

# CTC model (4 x BLSTMP)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight1.0|760|32771|80.1|13.0|6.8|2.3|22.2|98.7|
|decode_test_decode_ctcweight1.0_lm_word7184|760|32771|83.0|12.7|4.3|3.2|20.2|93.9|
|decode_train_dev_decode_ctcweight1.0|100|4007|82.6|12.0|5.4|1.7|19.1|99.0|
|decode_train_dev_decode_ctcweight1.0_lm_word7184|100|4007|84.1|12.3|3.5|2.2|18.1|99.0|
