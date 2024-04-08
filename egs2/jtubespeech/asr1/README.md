# The first result based on the single speaker split
- We used a CTC threshold -0.3
- The current scoring considers the space, and we need to remove them from scoring
## Environments
- date: `Fri Jun 11 11:19:26 EDT 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.10`
- pytorch version: `pytorch 1.7.1`
- Git hash: `5c44594ae531d9490f8106a4d81a8875fb361af2`
  - Commit date: `Thu Jun 10 00:24:57 2021 -0400`

## asr_train_asr_conformer2

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.best/valid_ss_th-0.3|2706|51430|89.0|5.7|5.4|5.4|16.5|68.7|
