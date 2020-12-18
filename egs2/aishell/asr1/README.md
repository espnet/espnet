# Conformer + specaug + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

## With Transformer LM
- Model link: https://zenodo.org/record/4105763#.X40xe2j7QUE
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_transformer_char_batch_bins2000000_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|95.7|4.2|0.1|0.1|4.4|33.7|
|decode_asr_rnn_lm_lm_train_lm_transformer_char_batch_bins2000000_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|95.4|4.5|0.1|0.1|4.7|35.0|

## With RNN LM
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|95.5|4.4|0.1|0.1|4.6|35.2|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|95.2|4.7|0.1|0.1|4.9|36.5|

## Without LM
- ASR config: [./conf/tuning/train_asr_conformer.yaml](./conf/tuning/train_asr_conformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_asr_model_valid.acc.ave/dev|14326|205341|95.6|4.3|0.1|0.1|4.5|35.0|
|decode_asr_rnn_asr_model_valid.acc.ave/test|7176|104765|95.2|4.7|0.1|0.1|4.9|36.7|

# Transformer + speed perturbation: feats=raw with same LM with the privious setting

I compared between `n_fft=512, hop_length=128`, `n_fft=400, hop_length=160`,  and `n_fft=512, hop_length=256`
with searching the best `batch_bins` to get the suitable configuration for each hop_length.

## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

## n_fft=512, hop_length=128
asr_train_asr_transformer2_raw_char_batch_typenumel_batch_bins8500000_optim_conflr0.0005_scheduler_confwarmup_steps30000_sp

- ASR config: [./conf/tuning/train_asr_transformer3.yaml](./conf/tuning/train_asr_transformer3.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)


### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.2|5.7|0.1|0.1|5.9|42.6|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.7|6.1|0.2|0.1|6.4|45.0|


## n_fft=400, hop_length=160
asr_train_asr_transformer2_raw_char_frontend_confn_fft400_frontend_confhop_length160_batch_typenumel_batch_bins6500000_optim_conflr0.0005_scheduler_confwarmup_steps30000_sp

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.1|5.7|0.1|0.1|6.0|43.0|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.5|6.3|0.2|0.1|6.6|45.4|

## n_fft=512, hop_length=256
asr_train_asr_transformer2_raw_char_frontend_confn_fft512_frontend_confhop_length256_batch_typenumel_batch_bins6000000_sp

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.9|0.1|0.1|6.1|43.5|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.3|6.5|0.2|0.1|6.8|45.8|


# Transformer + speed perturbation: feats=fbank_pitch, RNN-LM
Compatible setting with espnet1 to reproduce the previou result

## Environments
- date: `Fri Oct 16 11:10:17 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `20b0c89369d9dd3e05780b65fdd00a9b4f4891e5`
  - Commit date: `Mon Oct 12 09:28:20 2020 -0400`

- ASR config: [./conf/tuning/train_asr_transformer2.yaml](./conf/tuning/train_asr_transformer2.yaml)
- LM config: [./conf/tuning/train_lm_rnn2.yaml](./conf/tuning/train_lm_rnn2.yaml)

## asr_train_asr_transformer2_fbank_pitch_char_sp
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.9|0.1|0.1|6.1|43.4|
|decode_asr_rnn_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.4|6.4|0.2|0.1|6.7|45.9|

# The first result
## Environments
- date: `Sun Feb  2 02:03:55 CST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- pytorch version: `pytorch 1.1.0`
- Git hash: `e0fd073a70bcded6a0e6a3587630410a994ccdb8` (+ fixing https://github.com/espnet/espnet/pull/1533)
  - Commit date: `Sat Jan 11 06:09:24 2020 +0900`

## asr_train_asr_rnn_new_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_devdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|14326|205341|92.6|7.2|0.2|0.1|7.5|49.6|
|decode_testdecode_asr_rnn_lm_valid.loss.best_asr_model_valid.acc.best|7176|104765|91.6|8.2|0.3|0.2|8.6|53.4|

## asr_train_asr_transformer_lr0.002_fbank_pitch_char
### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|14326|205341|93.3|6.5|0.2|0.1|6.8|45.6|
|decode_test_decode_asr_rnn_lm_train_lm_char_valid.loss.best_asr_model_valid.acc.best|7176|104765|92.7|7.1|0.3|0.1|7.4|47.6|

