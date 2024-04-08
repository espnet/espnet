# RESULTS
## Environments
- date: `Fri Jul  9 20:43:31 PDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `049c1203da14ec06a8f8290575f5a44a5b1634d1`
  - Commit date: `Fri Jul 9 08:52:32 2021 -0700`

## asr_train_asr_raw_bpe1000
- ASR config: [conf/train_asr.yaml](conf/train_asr.yaml)
- Pretrained model: [https://zenodo.org/record/5090135](https://zenodo.org/record/5090135)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/sunda_test|2185|17916|98.5|1.2|0.3|0.1|1.6|5.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/sunda_test|2185|117265|99.5|0.2|0.3|0.1|0.6|5.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/sunda_test|2185|36414|98.5|0.9|0.6|0.2|1.6|5.0|
