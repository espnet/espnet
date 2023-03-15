# Streaming Conformer + specaug + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Mon Aug 23 16:31:48 CST 2021`
- python version: `3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.5.0`
- Git hash: `b94d07028099a80c9c690341981ae7d550b5ca24`
  - Commit date: `Mon Aug 23 00:47:47 2021 +0800`

## With Transformer LM
- Model link: (wait for upload)
- ASR config: [./conf/train_asr_streaming_cpnformer.yaml](./conf/train_asr_streaming_conformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|94.0|5.8|0.3|0.3|6.3|42.2|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|92.9|6.7|0.5|0.7|7.8|46.2|


# Streaming Transformer + speed perturbation: feats=raw, n_fft=512, hop_length=128
## Environments
- date: `Tue Aug 17 01:20:32 CST 2021`
- python version: `3.7.9 (default, Aug 31 2020, 12:42:55)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.5.0`
- Git hash: `6f5f848e0a9bfca1b73393779233bde34add3df1`
  - Commit date: `Mon Aug 16 21:50:08 2021 +0800`

## With Transformer LM
- Model link: (wait for upload)
- ASR config: [./conf/train_asr_streaming_transformer.yaml](./conf/train_asr_streaming_transformer.yaml)
- LM config: [./conf/tuning/train_lm_transformer.yaml](./conf/tuning/train_lm_transformer.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|14326|205341|93.6|6.2|0.1|0.5|6.8|46.8|
|decode_asr_streaming_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|7176|104765|93.0|6.7|0.2|0.8|7.8|50.7|


# E-Branchformer

## Environments
- date: `Sun Dec 18 12:21:46 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1`
- Git hash: `26f432bc859e5e40cac1a86042d498ba7baffbb0`
  - Commit date: `Fri Dec 9 02:16:01 2022 +0000`

## Without LM

- ASR config: [conf/tuning/train_asr_e_branchformer_e12_mlp1024_linear1024_mactrue_amp.yaml](conf/tuning/train_asr_e_branchformer_e12_mlp1024_linear1024_mactrue_amp.yaml)
- #Params: 37.88 M
- Model link: [https://huggingface.co/pyf98/aishell_e_branchformer](https://huggingface.co/pyf98/aishell_e_branchformer)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|14326|205341|95.9|4.0|0.1|0.1|4.2|33.1|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|7176|104765|95.6|4.3|0.1|0.1|4.5|34.6|




# Branchformer: initial

## Environments
- date: `Sun May 22 13:29:06 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.11.0`
- Git hash: `58a0a12ba48634841eb6616576d39e150239b4a2`
  - Commit date: `Sun May 22 12:49:35 2022 -0400`

## Without LM
- ASR config: [conf/tuning/train_asr_branchformer_e24_amp.yaml](conf/tuning/train_asr_branchformer_e24_amp.yaml)
- #Params: 45.43 M
- Model link: [https://huggingface.co/pyf98/aishell_branchformer_e24_amp](https://huggingface.co/pyf98/aishell_branchformer_e24_amp)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.4/dev|14326|205341|96.0|4.0|0.1|0.1|4.1|32.7|
|beam10_ctc0.4/test|7176|104765|95.7|4.2|0.1|0.1|4.4|34.1|



# Branchformer: using fast_selfattn

## Environments
- date: `Sat May 28 16:09:35 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: `69141f66a5f0ff3ca370f6abe5738d33819ff9ab`
  - Commit date: `Fri May 27 22:12:20 2022 -0400`

## Without LM
- ASR config: [conf/tuning/train_asr_branchformer_fast_selfattn_e24_amp.yaml](conf/tuning/train_asr_branchformer_fast_selfattn_e24_amp.yaml)
- #Params: 42.31 M
- Model link: [https://huggingface.co/pyf98/aishell_branchformer_fast_selfattn_e24_amp](https://huggingface.co/pyf98/aishell_branchformer_fast_selfattn_e24_amp)

### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.4/dev|14326|205341|95.8|4.1|0.1|0.1|4.3|33.3|
|beam10_ctc0.4/test|7176|104765|95.5|4.4|0.1|0.1|4.6|35.2|



# Conformer: new config

## Environments
- date: `Fri May 27 13:37:48 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.11.0`
- Git hash: `4f36236ed7c8a25c2f869e518614e1ad4a8b50d6`
  - Commit date: `Thu May 26 00:22:45 2022 -0400`

## Without LM
- ASR config: [conf/tuning/train_asr_conformer_e12_amp.yaml](conf/tuning/train_asr_conformer_e12_amp.yaml)
- #Params: 46.25 M
- Model link: [https://huggingface.co/pyf98/aishell_conformer_e12_amp](https://huggingface.co/pyf98/aishell_conformer_e12_amp)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam10_ctc0.4/dev|14326|205341|95.8|4.1|0.1|0.1|4.3|33.1|
|beam10_ctc0.4/test|7176|104765|95.4|4.4|0.1|0.1|4.6|34.7|



# E-Branchformer: CTC

## Environments
- date: `Sun Feb 19 13:24:02 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `8fa6361886c246afbd90c6e2ef98596628bdeaa8`
  - Commit date: `Fri Feb 17 16:47:46 2023 -0600`

## Without LM, beam size 1
- ASR config: [conf/tuning/train_asr_ctc_e_branchformer_e12.yaml](conf/tuning/train_asr_ctc_e_branchformer_e12.yaml)
- Params: 26.24M
- Model link: [https://huggingface.co/pyf98/aishell_ctc_e_branchformer_e12](https://huggingface.co/pyf98/aishell_ctc_e_branchformer_e12)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|14326|205341|94.7|5.2|0.1|0.1|5.4|40.9|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|7176|104765|94.2|5.7|0.1|0.1|6.0|43.0|



# Conformer: CTC

## Environments
- date: `Sun Feb 19 15:20:11 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `8fa6361886c246afbd90c6e2ef98596628bdeaa8`
  - Commit date: `Fri Feb 17 16:47:46 2023 -0600`

## Without LM, beam size 1
- ASR config: [conf/tuning/train_asr_ctc_conformer_e15_linear1024.yaml](conf/tuning/train_asr_ctc_conformer_e15_linear1024.yaml)
- Params: 26.76M
- Model link: [https://huggingface.co/pyf98/aishell_ctc_conformer_e15_linear1024](https://huggingface.co/pyf98/aishell_ctc_conformer_e15_linear1024)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|14326|205341|94.4|5.5|0.1|0.1|5.8|42.9|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|7176|104765|93.9|6.0|0.1|0.1|6.3|44.5|



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
