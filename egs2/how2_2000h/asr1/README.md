## End to End Speech Recognition

This recipe can be used to build E2E Speech Summarization models using restricted self-attention on the HowTo corpus of instructional videos. 

HowTo 2000h fbank-pitch features have been released to enable reproduction of this recipe. 

#Results on ASR


## asr_base_conformer_lf_mix
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|55215|93.1|4.8|2.1|1.9|8.8|56.7|
|decode_asr_model_valid.acc.best/held_out_test|2761|47348|92.7|5.0|2.3|2.2|9.5|54.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|276377|97.1|1.1|1.9|1.9|4.8|56.7|
|decode_asr_model_valid.acc.best/held_out_test|2761|236575|96.8|1.2|2.0|2.1|5.4|54.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.acc.best/dev5_test|3016|82484|94.1|3.5|2.4|2.2|8.0|56.7|
|decode_asr_model_valid.acc.best/held_out_test|2761|70264|93.9|3.7|2.4|2.7|8.9|54.6|
