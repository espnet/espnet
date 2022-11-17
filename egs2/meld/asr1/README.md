# MELD RESULTS
## Environments
- date: `Thu Nov 10 09:07:40 EST 2022`
- python version: `3.8.6 (default, Dec 17 2020, 16:57:01)  [GCC 10.2.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `a7bd6522b32ec6472c13f6a2289dcdff4a846c12`
  - Commit date: `Wed Sep 14 08:34:27 2022 -0400`

## asr_train_asr_hubert_transformer_adam_specaug_meld_raw_en_bpe850
- ASR config: conf/tuning/train_asr_hubert_transformer_adam_specaug_meld.yaml
- token_type: bpe
- keep_nbest_models: 5

|dataset|Snt|Emotion Classification (%)|
|---|---|---|
|decoder_asr_asr_model_valid.acc.ave_5best/test|2608|39.22|
|decoder_asr_asr_model_valid.acc.ave_5best/valid|1104|42.64|

### ASR results

#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decoder_asr_asr_model_valid.acc.ave_5best/test|2608|24809|55.5|28.0|16.5|8.4|52.9|96.5|
|decoder_asr_asr_model_valid.acc.ave_5best/valid|1104|10171|55.3|29.4|15.3|7.0|51.7|96.2|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decoder_asr_asr_model_valid.acc.ave_5best/test|2608|120780|71.1|10.7|18.2|10.6|39.5|96.5|
|decoder_asr_asr_model_valid.acc.ave_5best/valid|1104|49323|71.3|11.1|17.6|9.4|38.1|96.2|

#### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decoder_asr_asr_model_valid.acc.ave_5best/test|2608|35287|57.6|21.8|20.5|7.8|50.2|96.5|
|decoder_asr_asr_model_valid.acc.ave_5best/valid|1104|14430|57.4|23.2|19.4|6.1|48.6|96.2|


