# Lightweight Sinc Convolutions
## About Lightweight Sinc Convolutions

- [https://arxiv.org/abs/2010.07597](https://arxiv.org/abs/2010.07597)
- Usage of pyscripts/utils/plot_sinc_filters.py. Would you write? @lumaku


## [Sinc-BLSTMP with hop_size=240](conf/tuning/train_asr_sinc_rnn.yaml)
### Environments
- date: `Wed Nov 25 16:37:11 CET 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.4.0`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|13235|59.0|35.6|5.4|6.0|47.1|97.6|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|12990|57.9|35.7|6.3|5.6|47.7|98.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|79133|87.8|6.7|5.5|2.7|14.9|97.6|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|77966|86.9|7.0|6.2|2.5|15.6|98.2|



## [Sinc-Transformer with hop_size=200](conf/tuning/train_asr_sinc_transformer.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|13235|57.4|36.1|6.6|4.3|47.0|97.9|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|12990|55.6|37.1|7.3|4.5|48.9|98.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dt_it|1082|79133|87.5|6.4|6.1|2.4|14.8|97.9|
|decode_asr_asr_model_valid.acc.ave/et_it|1055|77966|87.1|6.6|6.3|2.4|15.3|98.3|

