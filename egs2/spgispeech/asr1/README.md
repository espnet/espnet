# RESULTS
## Environments
- date: `Thu Mar  4 09:32:06 EST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- pytorch version: `pytorch 1.7.1`
- Git hash: `fde26e9b249045a76e6cf809c85f9ab7118c2028`
  - Commit date: `Thu Mar 4 09:06:55 2021 -0500`

## unnormalizded text, bpe 5000 (asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000)
- https://zenodo.org/record/4585558
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_en_unnorm_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_4k_unnorm|4000|95631|94.9|4.5|0.6|0.5|5.5|61.0|
|decode_asr_lm_lm_train_lm_en_unnorm_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/val_unnorm|39341|948632|94.9|4.5|0.6|0.5|5.5|61.4|

## unnormalizded text, bpe 10000 (asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe10000)
- https://zenodo.org/record/4590907
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_en_unnorm_bpe10000_valid.loss.ave_asr_model_valid.acc.ave/dev_4k_unnorm|4000|95631|94.9|4.5|0.6|0.5|5.5|61.6|
|decode_asr_lm_lm_train_lm_en_unnorm_bpe10000_valid.loss.ave_asr_model_valid.acc.ave/val_unnorm|39341|948632|94.9|4.5|0.6|0.5|5.5|61.4|

## normalizded text, bpe 5000 (asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000)
- https://zenodo.org/record/4585546
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_4k|4000|95401|98.2|1.3|0.5|0.4|2.2|32.5|
|decode_asr_lm_lm_train_lm_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/val|39341|946469|98.1|1.3|0.5|0.4|2.3|33.6|

## normalizded text, bpe 10000 (asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe10000)
- https://zenodo.org/record/4585552
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_en_bpe10000_valid.loss.ave_asr_model_valid.acc.ave/dev_4k|4000|95401|97.7|1.6|0.7|0.6|2.9|38.7|
|decode_asr_lm_lm_train_lm_en_bpe10000_valid.loss.ave_asr_model_valid.acc.ave/val|39341|946469|97.6|1.7|0.8|0.6|3.0|39.2|
