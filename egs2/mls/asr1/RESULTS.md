# RESULTS
## Environments
- date: `Fri Jan 22 04:56:26 EST 2021`
- python version: `3.8.3 (default, May 19 2020, 18:47:26)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.6.0`
- Git hash: `c0c3724fe660abd205dbca9c9bbdffed1d2c79db`
  - Commit date: `Tue Jan 12 23:00:11 2021 -0500`

## asr_transformer_es
- lang: `es`
- data_split: `full`
- asr_config: `conf/train_asr.yaml`
- inference_config: `conf/decode_asr.yaml`
- model link: https://zenodo.org/record/4458452
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|88499|81.3|15.6|3.1|2.5|21.2|98.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|474976|94.3|2.9|2.7|1.4|7.1|98.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.best/es_test|2385|251160|88.6|7.9|3.5|2.1|13.6|98.6|


# RESULTS
## Environments
- date: `Wed Sep 13 23:39:31 EDT 2023`
- python version: `3.8.6 (default, Dec 17 2020, 16:57:01)  [GCC 10.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1+cu117`
- Git hash: `a8bc43b1bfc9518da7dd8be4cad0ef346ef222fc`
  - Commit date: `Sun Aug 20 16:17:23 2023 -0400`

## all 10h
- lang: `all`
- data_split: `10h`
- asr_config: `conf/tuning/train_asr_e_branchformer1_fbank_lre1-3.yaml`
- inference_config: `conf/tuning/decode_transformer_nolm.yaml`
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_de_test|3394|121689|69.0|26.8|4.3|3.6|34.7|99.8|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_en_test|3769|146611|54.6|39.9|5.4|4.6|50.0|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_es_test|2385|88499|77.6|18.7|3.7|2.9|25.3|99.5|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_fr_test|2426|93167|67.0|28.4|4.7|3.2|36.3|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_it_test|1262|40847|73.2|22.5|4.4|3.8|30.6|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_nl_test|3075|127722|67.1|28.5|4.4|4.2|37.1|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pl_test|520|17034|63.6|30.7|5.6|3.1|39.4|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pt_test|871|31255|63.5|30.3|6.2|4.0|40.5|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_de_test|3394|742421|92.0|4.0|4.0|2.3|10.3|99.8|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_en_test|3769|785323|82.4|10.3|7.2|4.6|22.1|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_es_test|2385|474976|94.6|2.9|2.5|1.6|7.0|99.5|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_fr_test|2426|531607|89.7|5.0|5.3|3.2|13.5|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_it_test|1262|230831|94.6|2.8|2.5|1.8|7.1|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_nl_test|3075|698026|91.9|4.1|4.0|3.3|11.4|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pl_test|520|111718|93.1|3.1|3.8|1.3|8.2|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pt_test|871|178026|89.8|5.5|4.7|2.5|12.7|100.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_de_test|3394|470137|85.8|9.8|4.4|2.1|16.3|99.8|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_en_test|3769|492873|71.9|20.7|7.4|4.7|32.8|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_es_test|2385|297162|89.5|7.3|3.2|1.6|12.1|99.5|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_fr_test|2426|347607|82.8|11.2|6.0|3.4|20.6|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_it_test|1262|146439|88.8|7.6|3.6|2.1|13.2|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_nl_test|3075|438029|85.2|10.7|4.1|3.2|18.0|99.9|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pl_test|520|82933|89.1|6.7|4.1|1.1|11.9|100.0|
|decode_asr_lm_lm_train_bpe150_valid.loss.ave_asr_model_valid.acc.ave/mls_pt_test|871|116658|82.5|11.7|5.7|2.9|20.3|100.0|
