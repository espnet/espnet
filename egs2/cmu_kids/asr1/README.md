# CMU Kids RECIPE

This is the recipe of the children speech recognition model with [CMU Kids dataset](https://catalog.ldc.upenn.edu/LDC97S63).

Before running the recipe, please download from https://catalog.ldc.upenn.edu/LDC97S63.
Then, edit 'CMU_KIDS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
CMU_KIDS=/path/to/cmu_kids

$ tree -L 2 /path/to/cmu_kids
/path/to/cmu_kids
└── cmu_kids
    ├── doc
    ├── kids
    └── table
    └── 0readme.1st
```


# References
[1] Maxine S. Eskenazi; KIDS: A database of children’s speech. J. Acoust. Soc. Am. 1 October 1996; 100: 2759. https://doi.org/10.1121/1.416340

# RESULTS
## Environments
- date: `Sun Jan 19 06:55:28 EST 2025`
- python version: `3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0]`
- espnet version: `espnet 202412`
- pytorch version: `pytorch 2.4.0`
- Git hash: `0fe7b8581fbc68841eb48776f052aa9a5989108c`
  - Commit date: `Tue Jan 14 20:06:15 2025 -0500`

## exp/asr_train_asr_wavlm_transformer_raw_en_bpe900_sp
Model: https://huggingface.co/wangpuupup/cmukids_wavlm_transformer_bpe900
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|6005|98.3|0.7|1.0|0.6|2.3|6.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|31847|98.7|0.3|1.1|0.6|1.9|6.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|9046|98.4|0.4|1.1|0.7|2.3|6.6|

## exp/asr_train_asr_wavlm_conformer_raw_en_bpe900_sp
Model: https://huggingface.co/wangpuupup/cmukids_wavlm_conformer_bpe900
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|6005|97.7|1.1|1.2|0.2|2.5|5.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|31847|98.2|0.5|1.4|0.3|2.1|5.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|9046|97.4|1.0|1.6|0.1|2.7|5.4|

## exp/asr_train_asr_wavlm_transformer_raw_en_bpe5000_sp
Model: https://huggingface.co/wangpuupup/cmukids_wavlm_transformer_bpe5000
bpe model is from espnet/myst_wavlm_aed_transformer
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|6005|98.5|0.6|0.8|0.3|1.8|4.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|31847|98.9|0.2|0.9|0.4|1.5|4.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|7212|98.5|0.6|0.9|0.4|1.9|4.8|

## exp/asr_train_asr_wavlm_conformer_raw_en_bpe5000_sp
Model: https://huggingface.co/wangpuupup/cmukids_wavlm_conformer_bpe5000
bpe model is from espnet/myst_wavlm_aed_transformer
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|6005|98.8|0.8|0.4|0.2|1.4|3.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|31847|99.1|0.3|0.5|0.2|1.1|3.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/test|754|7212|98.7|0.8|0.6|0.2|1.5|3.2|
