# E-Branchformer
## Environments
- date: `Mon Feb 20 21:37:20 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `8fa6361886c246afbd90c6e2ef98596628bdeaa8`
  - Commit date: `Fri Feb 17 16:47:46 2023 -0600`

## asr_train_asr_e_branchformer_e16_conv15_raw_jp_char_sp
- ASR config: [conf/tuning/train_asr_e_branchformer_e16_conv15.yaml](conf/tuning/train_asr_e_branchformer_e16_conv15.yaml)
- Params: 44.24M
- Model link: [https://huggingface.co/pyf98/jsut_e_branchformer](https://huggingface.co/pyf98/jsut_e_branchformer)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_train_lm_jp_char_valid.loss.ave_asr_model_valid.acc.ave/dev|250|6349|89.3|8.4|2.2|1.1|11.8|85.2|
|decode_transformer_lm_lm_train_lm_jp_char_valid.loss.ave_asr_model_valid.acc.ave/eval1|250|5928|88.2|9.6|2.2|1.1|13.0|88.0|


# conformer vs transformer results
## Environments
- date: `Wed Oct  7 20:04:20 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.4.0`
- Git hash: `53b60092f8d2b874f2e3f8d06244dc9d86949b2b`
  - Commit date: `Fri Oct 2 16:17:49 2020 -0400`

## asr_train_asr_conformer8_raw_char_sp (conformer)
- https://zenodo.org/record/4292742

### CER
#### 16k

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|250|6349|89.2|8.6|2.2|1.1|12.0|85.2|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/eval1|250|5928|87.5|9.9|2.6|1.3|13.9|86.4|


## asr_train_asr_conformer_raw_char_optim_conflr5_sp (conformer)
- https://zenodo.org/record/4073045

### CER
#### 16k

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|250|6349|87.4|10.3|2.4|1.4|14.0|88.8|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/eval1|250|5928|86.2|11.0|2.8|1.2|14.9|90.0|

## asr_train_asr_transformer_raw_char_sp (transformer)
- https://zenodo.org/record/4073054

### CER
#### 16k

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/dev|250|6349|85.8|11.0|3.2|1.6|15.8|91.2|
|decode_transformer_lm_lm_train_lm_char_valid.loss.ave_asr_model_valid.acc.ave/eval1|250|5928|84.1|12.5|3.5|1.7|17.6|93.6|

# Initial RNN results
## Environments
- date: `Thu Jan 16 01:20:43 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.6.0`
- pytorch version: `pytorch 1.0.1.post2`
- Git hash: `3b0f14e85e52ab5d178c412c26643759ef03d3fb`
  - Commit date: `Wed Jan 15 07:49:31 2020 +0900`

## asr_train_rnn_raw_char
### CER (S.Err for WER)

#### 16k

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_devdecode_rnn_lm_valid.loss.best_asr_model_valid.acc.best|250|6349|83.4|12.5|4.1|1.5|18.1|95.2|
|decode_eval1decode_rnn_lm_valid.loss.best_asr_model_valid.acc.best|250|5928|82.5|13.5|4.0|1.7|19.1|95.2|

#### 48k

Make sure to add following parameters.
```
--asr_args "--frontend_conf n_fft=$(echo ${fs} | python -c 'print(int(0.032 * float(input())))') --frontend_conf hop_length=$(echo ${fs} | python -c 'print(int(0.008 * float(input())))')"
```

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_devdecode_rnn_lm_valid.loss.best_asr_model_valid.acc.best|250|6349|84.7|12.0|3.3|1.7|17.0|94.8|
|decode_eval1decode_rnn_lm_valid.loss.best_asr_model_valid.acc.best|250|5928|83.5|13.1|3.4|1.6|18.0|94.8|
