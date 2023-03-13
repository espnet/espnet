# E-Branchformer
## Environments
- date: `Mon Dec 12 06:50:58 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1`
- Git hash: `26f432bc859e5e40cac1a86042d498ba7baffbb0`
  - Commit date: `Fri Dec 9 02:16:01 2022 +0000`

## asr_train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0_raw_en_bpe5000_sp

- ASR Config: [conf/tuning/train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml](conf/tuning/train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml)
- Params: 38.47M
- Model link: [https://huggingface.co/pyf98/librispeech_100_e_branchformer](https://huggingface.co/pyf98/librispeech_100_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|54402|94.6|5.0|0.3|0.8|6.1|55.4|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|50948|85.3|13.3|1.4|2.1|16.7|78.9|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|94.4|5.1|0.4|0.8|6.3|56.1|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|52343|85.0|13.6|1.4|2.0|17.0|80.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|288456|98.3|1.0|0.7|0.7|2.4|55.4|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|265951|93.6|4.0|2.4|2.0|8.3|78.9|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|281530|98.2|1.1|0.8|0.6|2.5|56.1|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|272758|93.7|3.8|2.5|1.9|8.2|80.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|69558|92.2|4.9|2.9|0.6|8.4|55.4|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|64524|81.9|12.8|5.2|2.3|20.4|78.9|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|66983|92.2|4.9|2.9|0.6|8.4|56.1|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|66650|81.5|13.0|5.5|2.2|20.7|80.3|


# E-Branchformer with CTC
## Environments
- date: `Sun Jan  1 15:05:07 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `b12e08c955276daa015cc40cf4f5977d87233db2`
  - Commit date: `Thu Dec 29 07:10:24 2022 -0500`

## asr_train_asr_ctc_e_branchformer_e12_raw_en_bpe5000_sp
- ASR config: [conf/tuning/train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml](conf/tuning/train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml)
- Params: 26.43M
- Model link: [https://huggingface.co/pyf98/librispeech_100_ctc_e_branchformer](https://huggingface.co/pyf98/librispeech_100_ctc_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_clean|2703|54402|91.8|7.5|0.7|1.0|9.2|70.1|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_other|2864|50948|80.4|17.4|2.2|2.8|22.4|87.8|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_clean|2620|52576|91.5|7.7|0.8|1.1|9.6|70.3|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_other|2939|52343|79.5|18.1|2.4|2.6|23.1|88.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_clean|2703|288456|97.1|1.2|1.7|1.1|4.0|70.1|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_other|2864|265951|91.2|4.5|4.3|3.0|11.8|87.8|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_clean|2620|281530|97.0|1.3|1.7|1.2|4.2|70.3|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_other|2939|272758|90.8|4.5|4.7|3.0|12.2|88.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_clean|2703|69558|89.6|5.9|4.5|0.9|11.3|70.1|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/dev_other|2864|64524|77.9|14.8|7.2|3.0|25.1|87.8|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_clean|2620|66983|89.6|6.0|4.4|1.0|11.4|70.3|
|decode_ctc_bs1_asr_model_valid.cer_ctc.ave/test_other|2939|66650|77.2|15.2|7.6|2.8|25.7|88.6|


# E-Branchformer: Transducer
## Environments
- date: `Sun Feb 19 01:32:24 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `8fa6361886c246afbd90c6e2ef98596628bdeaa8`
  - Commit date: `Fri Feb 17 16:47:46 2023 -0600`

## beam size: 10
- ASR config: [conf/tuning/train_asr_transducer_e_branchformer_e12_mlp1024_linear1024.yaml](conf/tuning/train_asr_transducer_e_branchformer_e12_mlp1024_linear1024.yaml)
- Params: 30.01M
- Model link: [https://huggingface.co/pyf98/librispeech_100_transducer_e_branchformer](https://huggingface.co/pyf98/librispeech_100_transducer_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|54402|94.0|5.3|0.6|0.7|6.6|58.6|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|50948|84.2|13.9|1.9|1.8|17.6|81.1|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|52576|93.9|5.4|0.7|0.7|6.8|58.5|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|52343|83.8|14.1|2.1|1.8|18.0|81.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|288456|97.8|1.1|1.1|0.7|2.8|58.6|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|265951|92.7|4.1|3.2|2.0|9.3|81.1|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|281530|97.8|1.1|1.1|0.7|2.9|58.5|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|272758|92.8|4.0|3.2|2.0|9.2|81.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|69558|91.6|5.2|3.2|0.7|9.1|58.6|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|64524|80.9|13.4|5.7|2.7|21.8|81.1|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|66983|91.6|5.2|3.2|0.8|9.2|58.5|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|66650|80.5|13.7|5.8|2.6|22.1|81.5|



# Conformer: Transducer
## Environments
- date: `Sun Feb 19 04:43:54 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `8fa6361886c246afbd90c6e2ef98596628bdeaa8`
  - Commit date: `Fri Feb 17 16:47:46 2023 -0600`

## beam_size: 10
- ASR config: [conf/tuning/train_asr_transducer_conformer_e15_linear1024.yaml](conf/tuning/train_asr_transducer_conformer_e15_linear1024.yaml)
- Params: 30.53M
- Model link: [https://huggingface.co/pyf98/librispeech_100_transducer_conformer](https://huggingface.co/pyf98/librispeech_100_transducer_conformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|54402|94.1|5.3|0.6|0.6|6.6|58.5|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|50948|83.9|14.0|2.1|1.8|17.9|81.4|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|52576|93.8|5.5|0.7|0.7|6.9|59.9|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|52343|83.7|14.1|2.2|1.7|18.1|81.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|288456|97.8|1.1|1.1|0.7|2.8|58.5|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|265951|92.5|4.2|3.3|2.0|9.4|81.4|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|281530|97.7|1.1|1.2|0.6|2.9|59.9|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|272758|92.6|4.0|3.4|1.9|9.3|81.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_transducer_asr_model_valid.loss.ave/dev_clean|2703|69558|91.7|5.2|3.1|0.7|9.1|58.5|
|decode_transducer_asr_model_valid.loss.ave/dev_other|2864|64524|80.4|13.7|5.8|2.6|22.1|81.4|
|decode_transducer_asr_model_valid.loss.ave/test_clean|2620|66983|91.5|5.2|3.3|0.7|9.2|59.9|
|decode_transducer_asr_model_valid.loss.ave/test_other|2939|66650|80.4|13.4|6.1|2.4|22.0|81.8|



# Conformer
## Environments
- date: `Mon Feb  7 21:28:00 EST 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.6a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `060fdb8b231b980c67b88a00fb8dd644aebbb1c0`
  - Commit date: `Mon Feb 7 21:26:51 2022 -0500`

## asr_conformer_win400_hop160_ctc0.3_lr2e-3_warmup15k_timemask5_amp_no-deterministic

GPU: a single V100-32GB

Training Time: 57072 seconds

Model: https://huggingface.co/pyf98/librispeech_100h_conformer


### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam1_ctc0.3/dev_clean|2703|54402|93.6|5.3|1.1|1.5|8.0|58.5|
|beam1_ctc0.3/dev_other|2864|50948|83.7|14.3|2.0|3.2|19.5|81.2|
|beam1_ctc0.3/test_clean|2620|52576|93.3|5.6|1.1|1.7|8.4|59.4|
|beam1_ctc0.3/test_other|2939|52343|83.5|14.4|2.1|2.9|19.4|83.3|
|beam20_ctc0.3/dev_clean|2703|54402|94.5|5.1|0.4|0.8|6.3|56.3|
|beam20_ctc0.3/dev_other|2864|50948|84.6|13.9|1.5|2.1|17.4|79.9|
|beam20_ctc0.3/test_clean|2620|52576|94.3|5.3|0.4|0.8|6.5|57.0|
|beam20_ctc0.3/test_other|2939|52343|84.7|13.7|1.6|2.0|17.3|81.6|
|timesync_beam20_ctc0.3/dev_clean|2703|54402|94.4|5.1|0.5|0.7|6.3|56.6|
|timesync_beam20_ctc0.3/dev_other|2864|50948|83.9|13.4|2.7|1.8|17.8|80.3|
|timesync_beam20_ctc0.3/test_clean|2620|52576|94.3|5.2|0.5|0.7|6.5|57.3|
|timesync_beam20_ctc0.3/test_other|2939|52343|84.1|13.4|2.4|1.8|17.7|82.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam1_ctc0.3/dev_clean|2703|288456|97.4|1.2|1.4|1.4|4.0|58.5|
|beam1_ctc0.3/dev_other|2864|265951|92.5|4.5|3.0|3.2|10.7|81.2|
|beam1_ctc0.3/test_clean|2620|281530|97.3|1.2|1.5|1.5|4.2|59.4|
|beam1_ctc0.3/test_other|2939|272758|92.6|4.3|3.1|2.9|10.3|83.3|
|beam20_ctc0.3/dev_clean|2703|288456|98.2|1.1|0.7|0.7|2.5|56.3|
|beam20_ctc0.3/dev_other|2864|265951|93.3|4.2|2.5|2.0|8.7|79.9|
|beam20_ctc0.3/test_clean|2620|281530|98.1|1.1|0.8|0.6|2.5|57.0|
|beam20_ctc0.3/test_other|2939|272758|93.5|4.0|2.6|1.9|8.4|81.6|
|timesync_beam20_ctc0.3/dev_clean|2703|288456|98.1|1.0|0.9|0.6|2.5|56.6|
|timesync_beam20_ctc0.3/dev_other|2864|265951|92.0|3.9|4.2|1.8|9.8|80.3|
|timesync_beam20_ctc0.3/test_clean|2620|281530|98.0|1.0|1.0|0.6|2.6|57.3|
|timesync_beam20_ctc0.3/test_other|2939|272758|92.5|3.7|3.8|1.7|9.3|82.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam1_ctc0.3/dev_clean|2703|69558|91.0|5.5|3.5|1.4|10.4|58.5|
|beam1_ctc0.3/dev_other|2864|64524|80.2|14.7|5.1|4.2|24.0|81.2|
|beam1_ctc0.3/test_clean|2620|66983|91.0|5.6|3.4|1.6|10.6|59.4|
|beam1_ctc0.3/test_other|2939|66650|80.0|14.4|5.6|3.7|23.7|83.3|
|beam20_ctc0.3/dev_clean|2703|69558|91.9|5.0|3.1|0.6|8.7|56.3|
|beam20_ctc0.3/dev_other|2864|64524|81.0|13.5|5.5|2.3|21.3|79.9|
|beam20_ctc0.3/test_clean|2620|66983|92.0|5.0|3.0|0.6|8.6|57.0|
|beam20_ctc0.3/test_other|2939|66650|81.2|13.0|5.8|2.0|20.9|81.6|
|timesync_beam20_ctc0.3/dev_clean|2703|69558|91.8|4.8|3.4|0.5|8.7|56.6|
|timesync_beam20_ctc0.3/dev_other|2864|64524|80.0|12.5|7.5|1.8|21.8|80.3|
|timesync_beam20_ctc0.3/test_clean|2620|66983|91.9|4.8|3.4|0.6|8.7|57.3|
|timesync_beam20_ctc0.3/test_other|2939|66650|80.4|12.2|7.4|1.8|21.4|82.2|


# Transformer
## Environments
- date: `Fri Feb 18 16:00:45 EST 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `f6779876103be2116de158a44757f8979eff0ab0`
  - Commit date: `Fri Feb 18 15:57:13 2022 -0500`

## asr_transformer_win400_hop160_ctc0.3_lr2e-3_warmup15k_timemask5_amp_no-deterministic

GPU: a single V100-32GB

Training Time: 42834 seconds

Model: https://huggingface.co/pyf98/librispeech_100h_transformer

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam20_ctc0.3/dev_clean|2703|54402|93.0|6.4|0.5|1.1|8.1|63.1|
|beam20_ctc0.3/dev_other|2864|50948|82.5|15.9|1.6|2.7|20.2|83.8|
|beam20_ctc0.3/test_clean|2620|52576|92.8|6.5|0.7|1.2|8.4|63.3|
|beam20_ctc0.3/test_other|2939|52343|82.1|16.0|1.9|2.6|20.5|84.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam20_ctc0.3/dev_clean|2703|288456|97.5|1.4|1.1|0.9|3.4|63.1|
|beam20_ctc0.3/dev_other|2864|265951|92.1|4.8|3.1|2.4|10.3|83.8|
|beam20_ctc0.3/test_clean|2620|281530|97.4|1.4|1.2|0.9|3.5|63.3|
|beam20_ctc0.3/test_other|2939|272758|92.0|4.7|3.2|2.3|10.2|84.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam20_ctc0.3/dev_clean|2703|69558|89.9|6.1|4.0|0.8|10.9|63.1|
|beam20_ctc0.3/dev_other|2864|64524|78.5|15.3|6.2|2.8|24.3|83.8|
|beam20_ctc0.3/test_clean|2620|66983|90.0|6.2|3.9|0.8|10.9|63.3|
|beam20_ctc0.3/test_other|2939|66650|77.9|15.2|6.9|2.5|24.6|84.8|


# RESULTS
## Environments
- date: `Mon Jan  9 23:06:34 CST 2023`
- python version: `3.9.13 (main, Aug 25 2022, 23:26:10)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `d89be931dcc8f61437ac49cbe39a773f2054c50c`
  - Commit date: `Mon Jan 9 11:06:45 2023 -0600`

## asr_whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs

- Huggingface model URL: https://huggingface.co/espnet/shihlun_asr_whisper_medium_finetuned_librispeech100

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dev_clean|2703|54798|97.7|1.9|0.3|0.3|2.6|30.1|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dev_other|2864|51528|95.3|4.3|0.4|0.6|5.3|45.4|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/test_clean|2620|53027|97.6|2.1|0.3|0.4|2.7|30.9|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/test_other|2939|52882|95.1|4.4|0.5|0.7|5.6|47.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dev_clean|2703|287287|99.3|0.3|0.4|0.3|1.0|30.1|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/dev_other|2864|265648|98.3|1.0|0.7|0.6|2.3|45.4|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/test_clean|2620|280691|99.3|0.3|0.3|0.3|1.0|30.9|
|decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave/test_other|2939|271738|98.3|1.0|0.7|0.7|2.4|47.5|
