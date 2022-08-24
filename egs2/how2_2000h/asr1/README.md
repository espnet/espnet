## End to End Speech Recognition with How2-2000h


HowTo 2000h fbank-pitch features have been released to enable reproduction of this recipe. 


# Results on ASR


## asr_base_conformer_lf_mix

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|55215|93.1|4.8|2.1|1.9|8.8|56.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|276377|97.1|1.1|1.9|1.9|4.8|56.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|82484|94.1|3.5|2.4|2.2|8.0|56.7|
