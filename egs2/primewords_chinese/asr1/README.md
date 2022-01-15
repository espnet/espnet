# Note
- Dataset: http://www.openslr.org/47/
- Please double check the data preparation stage when using this recipe in your own setting. Some processing might be inconsistent with other sources (if any). Currently we do ***not*** find a standard reference for data preparation, so the train/dev/test split is ***not*** "official". We do ***not*** include real speaker ids as well. Instead, utterance ids are used in `utt2spk`.


# Initial Experiment: Conformer + Speed Perturbation + SpecAugment, without LM

## Environments
- date: `Thu Dec 23 08:05:36 EST 2021`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.10.0`
- Git hash: `b687486451d28a3a3b5b04a23c10ebddcf09d13e`
  - Commit date: `Thu Dec 23 01:43:00 2021 -0500`

## asr_conformer_lr1e-3_warmup25k
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|7763|7763|15.3|84.7|0.0|0.0|84.7|84.7|
|decode_asr_asr_model_valid.acc.ave/test|3617|3617|16.2|83.8|0.0|0.0|83.8|83.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|7763|160698|84.2|15.3|0.5|0.5|16.3|84.7|
|decode_asr_asr_model_valid.acc.ave/test|3617|75151|84.7|14.9|0.4|0.4|15.7|83.8|
