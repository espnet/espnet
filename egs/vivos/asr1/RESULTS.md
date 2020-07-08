# mixed RNN/Transformer-Transducer (enc: VGG2L + 6 x Transformer, dec: 1 x LSTM)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|32771|83.3|10.7|5.9|2.7|19.4|94.9|
|decode_test_decode_default_lm|760|32771|85.6|10.3|4.1|2.7|17.1|94.2|
|decode_test_decode_default_lm_word7184|760|32771|85.8|10.1|4.1|2.7|16.9|93.7|
|decode_test_decode_nsc|760|32771|85.6|10.7|3.7|3.0|17.4|94.7|
|decode_test_decode_nsc_lm|760|32771|86.2|10.2|3.6|2.8|16.6|94.2|
|decode_test_decode_nsc_lm_word7184|760|32771|86.4|10.1|3.5|2.8|16.4|93.2|
|decode_train_dev_decode_default|100|4007|83.0|11.4|5.6|2.8|19.8|97.0|
|decode_train_dev_decode_default_lm|100|4007|85.9|10.8|3.3|2.6|16.8|94.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|86.2|10.8|3.0|2.3|16.1|91.0|
|decode_train_dev_decode_nsc|100|4007|86.1|11.4|2.5|2.8|16.7|96.0|
|decode_train_dev_decode_nsc_lm|100|4007|86.6|10.9|2.4|2.7|16.1|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.9|10.7|2.4|2.6|15.7|92.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|7722|61.2|36.2|2.5|0.4|39.2|94.9|
|decode_test_decode_default_lm|760|7722|64.9|33.8|1.3|0.5|35.6|94.2|
|decode_test_decode_default_lm_word7184|760|7722|66.0|32.7|1.3|0.5|34.5|93.7|
|decode_test_decode_nsc|760|7722|63.6|35.6|0.8|0.5|36.9|94.7|
|decode_test_decode_nsc_lm|760|7722|65.5|33.6|0.9|0.5|35.0|94.2|
|decode_test_decode_nsc_lm_word7184|760|7722|66.4|32.6|0.9|0.5|34.0|93.2|
|decode_train_dev_decode_default|100|927|60.1|37.1|2.8|0.4|40.3|97.0|
|decode_train_dev_decode_default_lm|100|927|64.7|34.3|1.0|0.3|35.6|94.0|
|decode_train_dev_decode_default_lm_word7184|100|927|65.7|33.5|0.8|0.3|34.6|91.0|
|decode_train_dev_decode_nsc|100|927|63.8|36.1|0.1|0.4|36.7|96.0|
|decode_train_dev_decode_nsc_lm|100|927|65.3|34.5|0.2|0.3|35.1|96.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|66.0|33.8|0.2|0.3|34.3|92.0|

# RNN-Transducer (enc: 4 x BLSTMP, dec: 1 x LSTM)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|32771|84.6|11.6|3.8|3.2|18.5|93.3|
|decode_test_decode_default_lm|760|32771|84.1|10.9|5.0|2.9|18.8|93.0|
|decode_test_decode_default_lm_word7184|760|32771|84.4|10.7|4.9|2.8|18.4|92.0|
|decode_test_decode_nsc|760|32771|85.2|11.4|3.4|3.4|18.2|93.0|
|decode_test_decode_nsc_lm|760|32771|85.5|11.1|3.4|3.3|17.8|92.6|
|decode_test_decode_nsc_lm_word7184|760|32771|85.9|10.8|3.3|3.2|17.4|92.0|
|decode_train_dev_decode_default|100|4007|84.9|11.6|3.5|2.2|17.3|97.0|
|decode_train_dev_decode_default_lm|100|4007|84.9|11.5|3.6|2.0|17.2|97.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|84.5|11.2|4.3|2.1|17.6|93.0|
|decode_train_dev_decode_nsc|100|4007|85.5|11.6|2.8|2.3|16.8|98.0|
|decode_train_dev_decode_nsc_lm|100|4007|86.0|11.3|2.7|2.4|16.4|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.3|10.9|2.8|2.2|15.9|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|7722|61.5|37.7|0.8|0.5|39.0|93.3|
|decode_test_decode_default_lm|760|7722|62.8|35.1|2.1|0.4|37.6|93.0|
|decode_test_decode_default_lm_word7184|760|7722|64.0|33.9|2.0|0.4|36.4|92.0|
|decode_test_decode_nsc|760|7722|62.2|37.2|0.6|0.6|38.4|93.0|
|decode_test_decode_nsc_lm|760|7722|63.6|35.7|0.7|0.5|37.0|92.6|
|decode_test_decode_nsc_lm_word7184|760|7722|64.8|34.5|0.7|0.5|35.7|92.0|
|decode_train_dev_decode_default|100|927|60.4|38.8|0.8|0.0|39.6|97.0|
|decode_train_dev_decode_default_lm|100|927|62.1|37.1|0.8|0.1|38.0|97.0|
|decode_train_dev_decode_default_lm_word7184|100|927|63.0|35.8|1.2|0.1|37.1|93.0|
|decode_train_dev_decode_nsc|100|927|62.0|37.9|0.1|0.1|38.1|98.0|
|decode_train_dev_decode_nsc_lm|100|927|63.1|36.8|0.1|0.0|36.9|97.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|64.8|35.0|0.2|0.1|35.3|96.0|

## RNN-Transducer w/ att (enc: 4 x BLSTMP, dec: 1 x LSTM)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|32771|83.4|11.9|4.8|3.2|19.8|95.4|
|decode_test_decode_default_lm|760|32771|84.0|11.2|4.8|2.8|18.8|92.4|
|decode_test_decode_default_lm_word7184|760|32771|84.1|11.0|4.9|2.7|18.6|92.1|
|decode_test_decode_nsc|760|32771|84.7|11.8|3.5|3.3|18.6|94.3|
|decode_test_decode_nsc_lm|760|32771|85.1|11.5|3.4|3.3|18.2|94.1|
|decode_test_decode_nsc_lm_word7184|760|32771|85.3|11.2|3.4|3.2|17.8|92.9|
|decode_train_dev_decode_default|100|4007|85.0|11.2|3.8|2.4|17.4|99.0|
|decode_train_dev_decode_default_lm|100|4007|84.9|11.2|4.0|1.9|17.1|95.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.6|10.4|4.0|1.8|16.2|95.0|
|decode_train_dev_decode_nsc|100|4007|85.8|11.4|2.8|2.2|16.4|96.0|
|decode_train_dev_decode_nsc_lm|100|4007|86.3|10.9|2.7|2.1|15.7|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|86.4|11.0|2.6|2.2|15.8|96.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|7722|59.0|39.3|1.7|0.5|41.5|95.4|
|decode_test_decode_default_lm|760|7722|61.6|36.6|1.8|0.4|38.8|92.4|
|decode_test_decode_default_lm_word7184|760|7722|62.7|35.3|2.0|0.4|37.6|92.1|
|decode_test_decode_nsc|760|7722|60.7|38.7|0.6|0.5|39.8|94.3|
|decode_test_decode_nsc_lm|760|7722|61.9|37.5|0.6|0.5|38.6|94.1|
|decode_test_decode_nsc_lm_word7184|760|7722|63.3|36.0|0.7|0.5|37.2|92.9|
|decode_train_dev_decode_default|100|927|61.3|37.8|1.0|0.0|38.7|99.0|
|decode_train_dev_decode_default_lm|100|927|63.0|35.5|1.5|0.0|37.0|95.0|
|decode_train_dev_decode_default_lm_word7184|100|927|65.3|33.8|1.0|0.0|34.7|95.0|
|decode_train_dev_decode_nsc|100|927|62.8|37.2|0.0|0.0|37.2|96.0|
|decode_train_dev_decode_nsc_lm|100|927|64.6|35.3|0.1|0.0|35.4|95.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|64.9|35.0|0.1|0.0|35.1|96.0|

## Transformer-Transducer (enc: VGG2L + 6 x TDNN-Transformer, dec: 2 x CausalConv1d-Transformer)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|32771|85.9|10.0|4.1|4.6|18.7|100.0|
|decode_test_decode_default_lm|760|32771|84.9|9.4|5.6|4.3|19.3|100.0|
|decode_test_decode_default_lm_word7184|760|32771|84.1|9.0|7.0|3.8|19.8|99.6|
|decode_test_decode_nsc|760|32771|86.2|9.9|3.9|4.6|18.4|100.0|
|decode_test_decode_nsc_lm|760|32771|86.4|9.5|4.1|4.5|18.0|100.0|
|decode_test_decode_nsc_lm_word7184|760|32771|86.5|9.4|4.1|4.4|17.9|100.0|
|decode_train_dev_decode_default|100|4007|89.0|8.9|2.1|4.1|15.1|100.0|
|decode_train_dev_decode_default_lm|100|4007|89.8|8.2|1.9|3.9|14.1|100.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|89.1|8.6|2.3|3.9|14.8|100.0|
|decode_train_dev_decode_nsc|100|4007|90.0|8.4|1.5|4.1|14.1|100.0|
|decode_train_dev_decode_nsc_lm|100|4007|90.0|8.4|1.6|4.1|14.1|100.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|90.0|8.4|1.6|4.0|14.0|100.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|7722|65.2|33.6|1.2|0.4|35.2|90.3|
|decode_test_decode_default_lm|760|7722|65.7|31.4|2.9|0.3|34.6|88.6|
|decode_test_decode_default_lm_word7184|760|7722|65.8|29.7|4.5|0.4|34.6|88.2|
|decode_test_decode_nsc|760|7722|65.9|32.9|1.2|0.4|34.6|90.5|
|decode_test_decode_nsc_lm|760|7722|66.8|31.7|1.5|0.4|33.6|89.6|
|decode_test_decode_nsc_lm_word7184|760|7722|67.6|30.8|1.6|0.4|32.8|88.9|
|decode_train_dev_decode_default|100|927|70.9|29.0|0.1|0.1|29.2|93.0|
|decode_train_dev_decode_default_lm|100|927|72.7|26.9|0.4|0.0|27.3|89.0|
|decode_train_dev_decode_default_lm_word7184|100|927|72.4|27.1|0.5|0.0|27.6|88.0|
|decode_train_dev_decode_nsc|100|927|72.0|28.0|0.0|0.0|28.0|90.0|
|decode_train_dev_decode_nsc_lm|100|927|72.3|27.7|0.0|0.0|27.7|90.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|73.0|27.0|0.0|0.0|27.0|86.0|

## Transformer-Transducer (enc: VGG2L + 8 x Transformer, dec: 2 x Transformer)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|32771|83.9|10.7|5.3|3.0|19.1|95.8|
|decode_test_decode_default_lm|760|32771|84.8|10.1|5.1|2.9|18.1|95.1|
|decode_test_decode_default_lm_word7184|760|32771|85.0|9.7|5.3|2.6|17.6|94.6|
|decode_test_decode_nsc|760|32771|84.6|10.8|4.6|3.3|18.6|95.9|
|decode_test_decode_nsc_lm|760|32771|85.4|10.1|4.5|3.1|17.6|95.1|
|decode_test_decode_nsc_lm_word7184|760|32771|85.6|9.8|4.6|2.9|17.3|94.1|
|decode_train_dev_decode_default|100|4007|85.1|11.4|3.5|2.6|17.5|98.0|
|decode_train_dev_decode_default_lm|100|4007|85.3|11.0|3.6|2.5|17.2|98.0|
|decode_train_dev_decode_default_lm_word7184|100|4007|85.4|10.8|3.9|2.4|17.0|98.0|
|decode_train_dev_decode_nsc|100|4007|85.2|11.8|3.0|2.6|17.4|98.0|
|decode_train_dev_decode_nsc_lm|100|4007|86.0|11.1|3.0|2.5|16.5|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|4007|85.8|10.8|3.3|2.3|16.4|97.0|

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_default|760|7722|57.6|40.0|2.4|0.5|42.9|95.8|
|decode_test_decode_default_lm|760|7722|60.7|36.8|2.5|0.5|39.7|95.1|
|decode_test_decode_default_lm_word7184|760|7722|62.5|34.6|2.8|0.5|37.9|94.6|
|decode_test_decode_nsc|760|7722|57.9|40.3|1.8|0.5|42.6|95.9|
|decode_test_decode_nsc_lm|760|7722|60.9|37.3|1.8|0.5|39.6|95.1|
|decode_test_decode_nsc_lm_word7184|760|7722|62.8|35.2|2.0|0.5|37.7|94.1|
|decode_train_dev_decode_default|100|927|59.3|39.6|1.1|0.1|40.8|98.0|
|decode_train_dev_decode_default_lm|100|927|61.3|37.4|1.3|0.0|38.7|98.0|
|decode_train_dev_decode_default_lm_word7184|100|927|63.2|35.3|1.5|0.0|36.8|98.0|
|decode_train_dev_decode_nsc|100|927|59.5|39.8|0.6|0.0|40.5|98.0|
|decode_train_dev_decode_nsc_lm|100|927|61.9|37.3|0.8|0.0|38.1|98.0|
|decode_train_dev_decode_nsc_lm_word7184|100|927|63.4|35.4|1.2|0.0|36.6|97.0|

# CTC model (4 x BLSTMP)

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_decode_ctcweight1.0|760|32771|80.1|13.0|6.8|2.3|22.2|98.7|
|decode_test_decode_ctcweight1.0_lm|760|32771|84.2|12.0|3.8|3.3|19.1|91.8|
|decode_test_decode_ctcweight1.0_lm_word7184|760|32771|83.0|12.7|4.3|3.2|20.2|93.9|
|decode_train_dev_decode_ctcweight1.0|100|4007|82.6|12.0|5.4|1.7|19.1|99.0|
|decode_train_dev_decode_ctcweight1.0_lm|100|4007|85.3|11.5|3.2|2.1|16.9|93.0|
|decode_train_dev_decode_ctcweight1.0_lm_word7184|100|4007|84.1|12.3|3.5|2.2|18.1|99.0|
