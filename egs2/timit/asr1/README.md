# RESULTS
## [Phone level] asr_train_asr_raw_word
LM doesn't improve the result.

### Environments
- date: `Sun Nov 22 05:36:34 JST 2020`
- python version: `3.8.5 (default, Aug  5 2020, 08:36:46)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.5`
- pytorch version: `pytorch 1.6.0`
- Git hash: `f157fcd651df34a714ad2bd5d97e46632e010096`
  - Commit date: `Sun Nov 22 04:29:17 2020 +0900`
- ASR config: [conf/tuning/train_asr_rnn.yaml](conf/tuning/train_asr_rnn.yaml)
- Decode config: [conf/tuning/decode_rnn.yaml](conf/tuning/decode_rnn.yaml)
- Pretrained model: https://zenodo.org/record/4284058

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|400|15057|84.4|11.2|4.4|2.3|17.9|99.0|
|decode_asr_asr_model_valid.acc.ave/test|192|7215|83.1|12.4|4.5|2.6|19.5|99.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|400|41742|91.0|4.3|4.8|2.8|11.9|99.0|
|decode_asr_asr_model_valid.acc.ave/test|192|20045|90.4|4.7|4.9|3.2|12.8|99.0|
