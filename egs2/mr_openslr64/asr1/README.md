# RESULTS
## Environments
- date: `Mon Mar 21 16:06:03 UTC 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.11.0+cu102`
- Git hash: `91325a1e58ca0b13494b94bf79b186b095fe0b58`
  - Commit date: `Mon Mar 21 00:40:52 2022 +0000`

## asr_train_asr_conformer_xlsr_raw_bpe150_sp

This recipe is for the Marathi language and is trained on the [OpenSLR Marathi](https://www.openslr.org/64/) multi-speaker speech data set.

The following results are obtained by using an XLSR frontend.

Train ASR Config: [conf/tuning/train_asr_conformer_xlsr.yaml](conf/tuning/train_asr_conformer_xlsr.yaml)

Trained Model: [espnet/marathi_openslr64](https://huggingface.co/espnet/marathi_openslr64)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_asr_model_valid.acc.ave/marathi_test|299|3625|72.9|22.5|4.7|1.7|28.9|88.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_asr_model_valid.acc.ave/marathi_test|299|20557|91.4|3.1|5.5|1.9|10.5|88.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_batch_size1_asr_model_valid.acc.ave/marathi_test|299|13562|86.5|6.3|7.1|1.4|14.9|88.6|
