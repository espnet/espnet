# Self-supervised learning features [HuBERT_large_ll60k, Conformer, utt_mvn](conf/tuning/train_asr_conformer7_hubert_960hr_large.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer2.yaml)

## Environments
- date: `Fri Aug  6 11:44:39 JST 2021`
- python version: `3.7.9 (default, Apr 23 2021, 13:48:31)  [GCC 5.5.0 20171010]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.7.0`
- Git hash: `0f7558a716ab830d0c29da8785840124f358d47b`
  - Commit date: `Tue Jun 8 15:33:49 2021 -0400`
- Pretrained model: https://huggingface.co/espnet/xuankai_chang_librispeech_asr_train_asr_conformer7_hubert_960hr_large_raw_en_bpe5000_sp_26epoch

### WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|54402|98.5|1.3|0.2|0.2|1.7|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|50948|96.8|2.8|0.4|0.3|3.4|33.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|52576|98.4|1.4|0.2|0.2|1.8|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|52343|96.8|2.8|0.4|0.4|3.6|36.0|

### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|288456|99.6|0.2|0.2|0.2|0.6|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|265951|98.8|0.6|0.6|0.3|1.5|33.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|281530|99.6|0.2|0.2|0.2|0.6|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|272758|98.9|0.5|0.5|0.4|1.4|36.0|

### TER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|68010|98.2|1.3|0.5|0.4|2.2|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|63110|96.0|2.8|1.2|0.6|4.6|33.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|65818|98.1|1.3|0.6|0.4|2.3|22.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|65101|96.0|2.7|1.3|0.6|4.6|36.0|


# Self-supervised learning features [Wav2Vec2_large_960hr, Conformer, utt_mvn](conf/tuning/train_asr_conformer7_wav2vec2_960hr_large.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer2.yaml)

## Environments
- date: `Sat Jul  3 23:10:19 JST 2021`
- python version: `3.7.9 (default, Apr 23 2021, 13:48:31)  [GCC 5.5.0 20171010]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.7.0`
- Git hash: `0f7558a716ab830d0c29da8785840124f358d47b`
  - Commit date: `Tue Jun 8 15:33:49 2021 -0400`
- Pretrained model: https://huggingface.co/espnet/xuankai_chang_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp_25epoch

### WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|54402|98.3|1.6|0.2|0.2|1.9|24.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|50948|95.1|4.3|0.6|0.4|5.4|42.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|52576|98.1|1.7|0.2|0.2|2.2|26.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|52343|95.3|4.1|0.6|0.5|5.2|45.8|

### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|288456|99.5|0.2|0.2|0.2|0.6|24.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|265951|98.1|1.0|0.9|0.5|2.4|42.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|281530|99.5|0.2|0.3|0.2|0.7|26.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|272758|98.3|0.8|0.9|0.5|2.3|45.8|

### TER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_clean|2703|68010|97.8|1.6|0.6|0.4|2.6|24.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/dev_other|2864|63110|94.1|4.3|1.6|1.1|7.0|42.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_clean|2620|65818|97.6|1.6|0.8|0.4|2.8|26.8|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.acc.best/test_other|2939|65101|94.3|4.0|1.8|1.0|6.7|45.8|


# Tuning warmup_steps
- Note
    - warmup_steps: 25000 -> 40000
    - lr: 0.0015 -> 0.0025

## Environments
- date: `Sat Mar 13 13:51:19 UTC 2021`
- python version: `3.8.8 (default, Feb 24 2021, 21:46:12)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.8`
- pytorch version: `pytorch 1.8.0`
- Git hash: `2ccd176da5de478e115600b874952cebc549c6ef`
  - Commit date: `Mon Mar 8 10:41:31 2021 +0000`

## With transformer LM
- ASR config: [conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml](conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://zenodo.org/record/4604066](https://zenodo.org/record/4604066)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.3|1.5|0.2|0.2|1.9|25.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.8|3.7|0.4|0.5|4.6|40.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.1|1.7|0.2|0.3|2.1|26.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|95.8|3.7|0.5|0.5|4.7|42.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|25.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.3|1.0|0.7|0.5|2.2|40.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.3|0.2|0.7|26.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|98.5|0.8|0.7|0.5|2.1|42.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|97.8|1.5|0.7|0.3|2.5|25.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|94.6|3.8|1.6|0.7|6.1|40.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|97.6|1.6|0.8|0.3|2.7|26.2|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|94.7|3.5|1.8|0.7|6.0|42.4|

## Without LM
- ASR config: [conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml](conf/tuning/train_asr_conformer7_n_fft512_hop_length256.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|54402|97.9|1.9|0.2|0.2|2.3|28.6|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|50948|94.5|5.1|0.5|0.6|6.1|48.3|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|52576|97.7|2.1|0.2|0.3|2.6|31.4|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|52343|94.7|4.9|0.5|0.7|6.0|49.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|288456|99.4|0.3|0.2|0.2|0.8|28.6|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|265951|98.0|1.2|0.8|0.7|2.7|48.3|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|281530|99.4|0.3|0.3|0.3|0.9|31.4|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|272758|98.2|1.0|0.7|0.7|2.5|49.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev_clean|2703|68010|97.5|1.9|0.7|0.4|2.9|28.6|
|decode_asr_asr_model_valid.acc.ave/dev_other|2864|63110|93.4|5.0|1.6|1.0|7.6|48.3|
|decode_asr_asr_model_valid.acc.ave/test_clean|2620|65818|97.2|2.0|0.8|0.4|3.3|31.4|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|65101|93.7|4.5|1.8|0.9|7.2|49.0|



# Updated the result of conformer with transformer LM
## Environments
- date: `Mon Feb 15 13:39:43 UTC 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.7.1`
- Git hash: `8eff1a983f6098111619328a7d8254974ae9dfca`
  - Commit date: `Wed Dec 16 02:22:50 2020 +0000`


## n_fft=512, hop_length=256
- ASR config: [conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml](conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://zenodo.org/record/4543018/](https://zenodo.org/record/4543018/)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.2|1.6|0.2|0.2|2.0|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.6|3.9|0.5|0.5|4.9|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.1|1.7|0.2|0.2|2.1|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|95.7|3.9|0.5|0.5|4.8|43.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.2|1.0|0.8|0.5|2.3|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.3|0.2|0.7|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|98.4|0.8|0.8|0.5|2.1|43.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/test_other|2939|65101|93.3|4.8|1.8|1.0|7.6|52.5|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|97.7|1.6|0.7|0.3|2.7|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|94.3|3.9|1.7|0.8|6.4|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|97.5|1.6|0.9|0.3|2.8|26.3|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|94.6|3.6|1.8|0.6|6.1|43.8|

## n_fft=400, hop_length=160
- ASR config: [conf/tuning/train_asr_conformer6_n_fft400_hop_length160.yaml](conf/tuning/train_asr_conformer6_n_fft400_hop_length160.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://zenodo.org/record/4543003](https://zenodo.org/record/4543003)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.2|1.6|0.2|0.2|2.0|25.5|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.6|3.9|0.5|0.5|4.9|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.1|1.7|0.2|0.3|2.1|26.4|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|95.5|4.0|0.5|0.6|5.1|44.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|25.5|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.3|1.0|0.8|0.5|2.3|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.3|0.2|0.7|26.4|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|98.4|0.9|0.8|0.5|2.2|44.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|97.7|1.6|0.7|0.3|2.6|25.5|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|94.3|3.9|1.7|0.7|6.4|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|97.5|1.6|0.8|0.3|2.7|26.4|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|94.3|3.8|1.9|0.6|6.3|44.1|

## n_fft=512, hop_length=128
- ASR config: [conf/tuning/train_asr_conformer6_n_fft512_hop_length128.yaml](conf/tuning/train_asr_conformer6_n_fft512_hop_length128.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://zenodo.org/record/4541452](https://zenodo.org/record/4541452)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.2|1.6|0.2|0.2|2.0|26.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.6|3.9|0.5|0.5|4.9|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.1|1.7|0.2|0.3|2.1|26.1|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|95.5|4.0|0.6|0.6|5.1|44.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|26.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.2|1.0|0.8|0.5|2.3|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.3|0.2|0.2|0.7|26.1|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|98.3|0.9|0.8|0.6|2.2|44.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|97.7|1.6|0.7|0.3|2.6|26.0|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|94.3|3.9|1.7|0.8|6.5|41.6|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|97.6|1.6|0.8|0.3|2.7|26.1|
|decode_asr_lm_lm_train_lm_transformer2_bpe5000_scheduler_confwarmup_steps25000_batch_bins500000000_accum_grad2_use_amptrue_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|94.3|3.7|2.0|0.7|6.4|44.4|


# The first conformer result
## Environments
- date: `Mon Nov 16 18:59:34 EST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.2`
- pytorch version: `pytorch 1.4.0`
- Git hash: `eda997f9e97ad959c6b13df1b34eb24fb8c52768`
  - Commit date: `Thu Oct 8 07:32:49 2020 -0400`

## asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp
- https://zenodo.org/record/4276519
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.0|1.8|0.2|0.3|2.3|27.6|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.0|4.3|0.6|0.5|5.5|45.1|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|28.2|
|decode_asr_lm_lm_train_lm_adam_bpe_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|94.9|4.5|0.7|0.6|5.8|48.6|

# The second tentative result
## Environments
- date: `Mon Jul 20 04:38:33 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `7b37518e0c7bc0c550a80b83fff4b3e927ebb142`
  - Commit date: `Wed Jul 8 17:25:03 2020 -0400`

- Update from the previous result
  - ASR: dropout 0.1, encoder 18 layers, subsampling (2x3=6), and speed perturbation
  - LM: Adam optimizer

## asr_train_asr_transformer_e18_raw_bpe_sp
- https://zenodo.org/record/3966501
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2703|54402|97.9|1.9|0.2|0.2|2.3|28.5|
|decode_dev_other_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2864|50948|94.7|4.6|0.6|0.6|5.9|46.0|
|decode_test_clean_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2620|52576|97.7|2.0|0.3|0.3|2.5|30.0|
|decode_test_other_decode_asr_beam_size20_lm_train_lm_adam_bpe_valid.loss.best_asr_model_valid.acc.best|2939|52343|94.5|4.8|0.7|0.7|6.2|50.0|


# The first tentative result
## Environments
- date: `Fri Jul  3 04:48:06 EDT 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `af9cb2449a15b89490964b413a9a02422a26fa5e`
  - Commit date: `Thu Jul 2 07:56:03 2020 -0400`

- beam size 20 (60 in the best system), ASR epoch ~60, LM epoch 2

## asr_train_asr_transformer_raw_bpe_optim_conflr0.001
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_dev_clean_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2703|54402|97.2|2.4|0.3|0.3|3.1|35.2|
|decode_dev_other_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2864|50948|93.2|6.0|0.8|0.8|7.6|54.7|
|decode_test_clean_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2620|52576|97.0|2.6|0.4|0.4|3.4|37.4|
|decode_test_other_decode_asr_beam_size20_lm_train_lm_bpe_valid.loss.best_asr_model_valid.acc.best|2939|52343|92.9|6.1|1.0|0.9|8.0|58.3|
