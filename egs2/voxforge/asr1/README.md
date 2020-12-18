- [Lightweight Sinc Convolutions](./README_LightweightSincConvs.md)

# The second results
## Environments
- date: `Mon Feb 17 18:11:26 JST 2020`
- python version: `3.7.5 (default, Oct 25 2019, 15:51:11)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.4`
- pytorch version: `pytorch 1.4.0`
- Git hash: `89cc53646f3afb634c2bf10b4a69ddd368479b74`
  - Commit date: `Thu Jan 23 19:09:29 2020 +0900

## [200 epoch Transformer feats_type=raw](conf/tuning/train_asr_transformer_2.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|13235|68.1|26.3|5.6|3.8|35.7|95.6|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|12990|69.3|25.7|5.0|3.9|34.6|93.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|79133|92.4|4.1|3.5|2.0|9.6|95.6|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|77966|92.7|3.9|3.4|1.8|9.1|93.9|

## [200 epoch Transformer feats_type=fbank_pitch](conf/tuning/train_asr_transformer_2.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|13235|67.9|26.9|5.2|4.1|36.2|95.6|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|12990|68.3|26.6|5.1|4.4|36.1|95.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|79133|92.2|4.2|3.6|2.0|9.9|95.6|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|77966|92.4|4.0|3.6|1.9|9.5|95.7|


## [VGG-BLSTMP feats_type=raw](conf/tuning/train_asr_rnn_2.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|13235|63.1|32.0|4.9|5.1|42.0|96.9|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|12990|63.7|31.3|5.0|5.3|41.7|96.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|79133|90.8|5.2|4.0|2.5|11.7|96.9|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|77966|90.8|5.1|4.0|2.3|11.4|96.9|


## [VGG-BLSTMP feats_type=fbank_pitch](conf/tuning/train_asr_rnn_2.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|13235|63.0|32.1|4.9|5.1|42.1|97.4|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|12990|63.3|31.8|4.9|5.5|42.2|97.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|79133|90.8|5.2|4.0|2.6|11.8|97.4|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|77966|90.7|5.2|4.1|2.4|11.6|97.3|


# The first results
## Environments
- date: `Thu Jan 23 19:10:23 JST 2020`
- python version: `3.7.5 (default, Oct 25 2019, 15:51:11)  [GCC 7.3.0]`
- espnet version: `espnet 0.5.4`
- pytorch version: `pytorch 1.4.0`
- Git hash: `89cc53646f3afb634c2bf10b4a69ddd368479b74`
  - Commit date: `Thu Jan 23 19:09:29 2020 +0900`

## asr_train_asr_transformer_raw_char_normalize_confnorm_varsFalse
### WER (valid.acc.ave)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|13235|62.4|31.4|6.1|3.7|41.3|97.2|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|12990|61.6|31.9|6.5|3.9|42.3|97.5|

### WER (valid.acc.best)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|13235|54.1|38.4|7.5|4.1|50.0|99.1|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|12990|53.4|38.4|8.3|4.1|50.7|98.8|


### CER (valid.acc.ave)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.ave|1082|79133|91.1|4.4|4.5|2.0|10.9|97.2|
|decode_et_itdecode_asr_asr_model_valid.acc.ave|1055|77966|91.0|4.5|4.6|1.9|10.9|97.5|


### CER (valid.acc.best)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|79133|88.4|5.5|6.1|2.5|14.0|99.1|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|77966|88.3|5.5|6.2|2.3|14.0|98.8|


## VGG-GRU ( asr_train_asr_rnn_raw_char_normalize_confnorm_varsFalse )
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|13235|60.1|34.7|5.2|5.1|45.0|97.6|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|12990|60.3|34.3|5.4|5.5|45.2|97.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dt_itdecode_asr_asr_model_valid.acc.best|1082|79133|90.1|5.5|4.4|2.5|12.4|97.6|
|decode_et_itdecode_asr_asr_model_valid.acc.best|1055|77966|90.1|5.6|4.3|2.4|12.3|97.2|
