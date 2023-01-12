## End to End Speech Recognition with How2-2000h


# Data Download and Preparation
HowTo 2000h fbank-pitch features have been released to enable reproduction of this recipe. 

You can request the use of this data using our (data request form)[https://docs.google.com/forms/d/e/1FAIpQLSfW2i8UnjuoH2KKSU0BvcKRbhnk_vL3HcNlM0QLsJGb_UEDVQ/viewform]

For ASR and Summarization, please request the data labeled "(audio_2000) fbank+pitch features in Kaldi scp/ark format for 2000 hours"

You will recieve a data download link shortly after you submit the form.
Then you can prepare the data directory by providing your link as follows:


```bash
./run.sh --local_data_opts "--data_url <insert-link-here>"
```


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
