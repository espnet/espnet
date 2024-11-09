# E-Branchformer
- Params: 148.92 M
- ASR config: [conf/tuning/train_asr_e_branchformer.yaml](conf/tuning/train_asr_e_branchformer.yaml)
- Model link: [https://huggingface.co/espnet/libriheavy_small_ebranchformer](https://huggingface.co/espnet/libriheavy_small_ebranchformer)

# RESULTS
## Environments
- date: `Fri Oct 18 10:37:57 WEST 2024`
- python version: `3.10.14 | packaged by conda-forge | (main, Mar 20 2024, 12:45:18) [GCC 12.3.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `f6f011d328fb877b098321975280cadf8c64247a`
  - Commit date: `Tue Apr 9 01:44:27 2024 +0000`

## exp/asr_train_asr_e_branchformer_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2557|102701|96.0|3.3|0.7|0.6|4.6|67.5|
|decode_asr_asr_model_valid.acc.ave/test_other|2815|111836|90.8|7.1|2.0|1.1|10.2|82.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2557|533368|98.6|0.6|0.8|0.5|1.9|67.5|
|decode_asr_asr_model_valid.acc.ave/test_other|2815|581017|96.4|1.6|2.0|1.1|4.7|82.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_clean|2557|127083|94.7|3.3|1.9|0.7|5.9|67.5|
|decode_asr_asr_model_valid.acc.ave/test_other|2815|144295|88.3|6.6|5.2|1.3|13.0|82.9|

## exp/asr_train_asr_e_branchformer_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|5348|218645|93.3|5.3|1.4|0.8|7.5|76.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|5348|1137810|97.5|1.1|1.4|0.8|3.3|76.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|org/dev|5348|277457|91.4|5.0|3.7|1.0|9.6|76.3|
