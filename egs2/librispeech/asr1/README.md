# Conformer-RNN Transducer

## Environments
- date: `Wed Apr 27 09:30:57 EDT 2022`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `21d19be00089678ca27f7fce474ef8d787689512`
  - Commit date: `Wed Mar 16 08:06:52 2022 -0400`
- ASR config: [conf/tuning/transducer/train_conformer-rnn_transducer.yaml](conf/tuning/transducer/train_conformer-rnn_transducer.yaml)
- Decode config: [conf/tuning/transducer/decode.yaml](conf/tuning/transducer/decode.yaml)
- Pretrained model: [https://huggingface.co/espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp](https://huggingface.co/espnet/chai_librispeech_asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp)

## asr_train_conformer-rnn_transducer_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|97.7|2.1|0.2|0.3|2.6|31.5|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_other|2864|50948|93.8|5.6|0.6|0.6|6.8|50.8|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_clean|2620|52576|97.5|2.3|0.2|0.3|2.8|32.7|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_other|2939|52343|94.1|5.3|0.6|0.7|6.6|51.8|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|98.0|1.8|0.2|0.2|2.2|28.2|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_other|2864|50948|94.8|4.5|0.7|0.5|5.7|45.1|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|29.3|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_other|2939|52343|94.9|4.3|0.7|0.5|5.6|47.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|99.4|0.4|0.3|0.2|0.9|31.5|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_other|2864|265951|97.7|1.4|0.9|0.8|3.0|50.8|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_clean|2620|281530|99.4|0.4|0.3|0.3|0.9|32.7|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_other|2939|272758|97.9|1.2|0.9|0.8|2.8|51.8|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|99.4|0.3|0.3|0.2|0.8|28.2|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_other|2864|265951|97.9|1.1|1.0|0.6|2.7|45.1|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_clean|2620|281530|99.4|0.3|0.3|0.2|0.9|29.3|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_other|2939|272758|98.1|0.9|1.0|0.6|2.5|47.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_clean|2703|68010|97.2|2.1|0.7|0.4|3.3|31.5|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/dev_other|2864|63110|92.7|5.6|1.7|1.2|8.6|50.8|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_clean|2620|65818|97.0|2.2|0.9|0.4|3.4|32.7|
|decode_lm_weight0.0_asr_model_valid.loss.ave_10best/test_other|2939|65101|93.0|5.1|1.9|1.0|8.0|51.8|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_clean|2703|68010|97.5|1.8|0.8|0.4|2.9|28.2|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/dev_other|2864|63110|93.5|4.5|1.9|0.9|7.4|45.1|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_clean|2620|65818|97.3|1.9|0.8|0.4|3.0|29.3|
|decode_lm_weight0.4_lm_lm_train_lm_transformer2_en_bpe5000_17epoch_asr_model_valid.loss.ave_10best/test_other|2939|65101|93.9|4.1|1.9|0.8|6.9|47.0|

# Self-supervised learning features [HuBERT_large_ll60k, Conformer, utt_mvn](conf/tuning/train_asr_conformer7_hubert_ll60k_large.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer2.yaml)

## Environments
- date: `Sat Jan  1 23:24:39 EST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.8.1`
- Git hash: `37a5c7cdb84b1d2361f4a4fa08826b2873bf7753`
  - Commit date: `Thu Nov 25 05:30:02 2021 +0000`
- Pretrained model: https://huggingface.co/espnet/simpleoier_librispeech_asr_train_asr_conformer7_hubert_ll60k_large_raw_en_bpe5000_sp

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.5|1.4|0.2|0.2|1.7|22.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|96.7|3.0|0.3|0.3|3.6|36.0|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.4|1.4|0.2|0.2|1.8|23.4|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|96.6|3.1|0.3|0.4|3.7|37.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.7|0.2|0.2|0.2|0.5|22.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|99.0|0.6|0.5|0.4|1.4|36.0|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.7|0.2|0.2|0.2|0.5|23.4|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|99.1|0.5|0.4|0.4|1.3|37.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|98.2|1.4|0.4|0.4|2.2|22.9|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|96.1|3.0|0.9|0.8|4.7|36.0|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|98.1|1.4|0.5|0.4|2.3|23.4|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|96.1|2.9|1.1|0.7|4.6|37.2|

# Self-supervised learning features [WavLM_large, Conformer, utt_mvn](conf/tuning/train_asr_conformer7_wavlm_large.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer2.yaml)

## Environments
- date: `Tue Jan  4 20:52:48 EST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.8.1`
- Git hash: `37a5c7cdb84b1d2361f4a4fa08826b2873bf7753`
  - Commit date: `Thu Nov 25 05:30:02 2021 +0000`
- Pretrained model: https://huggingface.co/espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.4|1.4|0.1|0.2|1.7|23.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|96.7|3.0|0.3|0.3|3.6|35.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.4|1.5|0.1|0.2|1.8|23.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|96.7|3.0|0.3|0.4|3.7|37.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.7|0.2|0.2|0.2|0.5|23.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.9|0.6|0.4|0.4|1.5|35.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.6|0.2|0.2|0.2|0.6|23.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|99.1|0.5|0.4|0.4|1.3|37.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|98.2|1.4|0.4|0.3|2.1|23.1|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|96.0|3.1|0.9|0.9|4.9|35.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|98.1|1.4|0.5|0.4|2.3|23.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|96.1|2.9|1.0|0.8|4.7|37.9|

# Self-supervised learning features [Wav2Vec2_large_960hr, Conformer, utt_mvn](conf/tuning/train_asr_conformer7_wav2vec2_960hr_large.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer2.yaml)

## Environments
- date: `Thu Dec 16 23:20:01 EST 2021`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.8.1`
- Git hash: `37a5c7cdb84b1d2361f4a4fa08826b2873bf7753`
  - Commit date: `Thu Nov 25 05:30:02 2021 +0000`
- Pretrained model: https://huggingface.co/espnet/simpleoier_librispeech_asr_train_asr_conformer7_wav2vec2_960hr_large_raw_en_bpe5000_sp

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|54402|98.3|1.6|0.2|0.2|1.9|24.6|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|50948|95.2|4.3|0.5|0.5|5.2|42.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|52576|98.1|1.6|0.2|0.2|2.1|25.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|52343|95.4|4.1|0.5|0.5|5.1|45.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|288456|99.6|0.2|0.2|0.2|0.6|24.6|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|265951|98.3|1.0|0.7|0.6|2.3|42.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|281530|99.5|0.2|0.3|0.2|0.7|25.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|272758|98.4|0.8|0.7|0.6|2.1|45.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_clean|2703|68010|98.0|1.5|0.5|0.4|2.4|24.6|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/dev_other|2864|63110|94.4|4.3|1.3|1.2|6.8|42.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_clean|2620|65818|97.7|1.6|0.7|0.4|2.7|25.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/test_other|2939|65101|94.5|3.9|1.5|1.0|6.4|45.1|




# Branchformer, `hope_length=160, num_blocks=18, cgmlp_linear_units=3072`
- Params: 116.88 M
- ASR config: [conf/tuning/train_asr_branchformer_hop_length160_e18_linear3072.yaml](conf/tuning/train_asr_branchformer_hop_length160_e18_linear3072.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Model link: [https://huggingface.co/pyf98/librispeech_branchformer_e18_linear3072](https://huggingface.co/pyf98/librispeech_branchformer_e18_linear3072)

# RESULTS
## Environments
- date: `Fri Jun  3 02:25:27 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202205`
- pytorch version: `pytorch 1.11.0`
- Git hash: `415e7ac5d1ca92ef0d91510086614899139b1e8f`
  - Commit date: `Mon May 30 23:48:29 2022 -0400`

## asr_train_asr_branchformer_hop_length160_e18_linear3072_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|till60epoch_beam60_ctc0.3/dev_clean|2703|54402|98.1|1.7|0.2|0.2|2.1|26.7|
|till60epoch_beam60_ctc0.3/dev_other|2864|50948|95.3|4.4|0.3|0.5|5.2|43.7|
|till60epoch_beam60_ctc0.3/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|28.1|
|till60epoch_beam60_ctc0.3/test_other|2939|52343|95.3|4.3|0.4|0.6|5.3|45.8|
|till60epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.4|1.4|0.2|0.2|1.7|23.0|
|till60epoch_beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.4|3.3|0.3|0.4|4.0|36.3|
|till60epoch_beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.3|1.5|0.2|0.3|2.0|23.9|
|till60epoch_beam60_ctc0.3_lm0.6/test_other|2939|52343|96.3|3.3|0.4|0.5|4.2|39.2|
|till70epoch_beam60_ctc0.3/dev_clean|2703|54402|98.1|1.7|0.2|0.2|2.1|27.0|
|till70epoch_beam60_ctc0.3/dev_other|2864|50948|95.4|4.3|0.4|0.5|5.1|43.2|
|till70epoch_beam60_ctc0.3/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.3|27.8|
|till70epoch_beam60_ctc0.3/test_other|2939|52343|95.4|4.2|0.4|0.6|5.3|45.3|
|till70epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.4|1.4|0.2|0.2|1.8|23.8|
|till70epoch_beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.4|3.3|0.3|0.4|4.0|36.6|
|till70epoch_beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.3|1.6|0.2|0.2|2.0|24.0|
|till70epoch_beam60_ctc0.3_lm0.6/test_other|2939|52343|96.3|3.3|0.5|0.5|4.2|39.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|till60epoch_beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|26.7|
|till60epoch_beam60_ctc0.3/dev_other|2864|265951|98.3|1.0|0.6|0.6|2.3|43.7|
|till60epoch_beam60_ctc0.3/test_clean|2620|281530|99.5|0.3|0.2|0.2|0.8|28.1|
|till60epoch_beam60_ctc0.3/test_other|2939|272758|98.5|0.9|0.6|0.6|2.2|45.8|
|till60epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.6|0.2|0.2|0.2|0.6|23.0|
|till60epoch_beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.6|0.9|0.6|0.5|1.9|36.3|
|till60epoch_beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.5|0.2|0.2|0.2|0.7|23.9|
|till60epoch_beam60_ctc0.3_lm0.6/test_other|2939|272758|98.7|0.7|0.6|0.5|1.8|39.2|
|till70epoch_beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|27.0|
|till70epoch_beam60_ctc0.3/dev_other|2864|265951|98.3|1.0|0.6|0.6|2.2|43.2|
|till70epoch_beam60_ctc0.3/test_clean|2620|281530|99.5|0.3|0.2|0.2|0.8|27.8|
|till70epoch_beam60_ctc0.3/test_other|2939|272758|98.5|0.9|0.6|0.7|2.1|45.3|
|till70epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.5|0.2|0.2|0.2|0.6|23.8|
|till70epoch_beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.6|0.8|0.6|0.5|1.9|36.6|
|till70epoch_beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.5|0.2|0.2|0.2|0.7|24.0|
|till70epoch_beam60_ctc0.3_lm0.6/test_other|2939|272758|98.6|0.7|0.6|0.5|1.9|39.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|till60epoch_beam60_ctc0.3/dev_clean|2703|68010|97.7|1.7|0.6|0.3|2.7|26.7|
|till60epoch_beam60_ctc0.3/dev_other|2864|63110|94.2|4.4|1.4|0.9|6.7|43.7|
|till60epoch_beam60_ctc0.3/test_clean|2620|65818|97.4|1.8|0.8|0.3|3.0|28.1|
|till60epoch_beam60_ctc0.3/test_other|2939|65101|94.2|4.1|1.7|0.7|6.5|45.8|
|till60epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|68010|98.0|1.4|0.6|0.3|2.3|23.0|
|till60epoch_beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.2|3.5|1.3|0.6|5.4|36.3|
|till60epoch_beam60_ctc0.3_lm0.6/test_clean|2620|65818|97.7|1.5|0.8|0.3|2.5|23.9|
|till60epoch_beam60_ctc0.3_lm0.6/test_other|2939|65101|95.2|3.1|1.7|0.6|5.3|39.2|
|till70epoch_beam60_ctc0.3/dev_clean|2703|68010|97.7|1.7|0.6|0.4|2.7|27.0|
|till70epoch_beam60_ctc0.3/dev_other|2864|63110|94.2|4.4|1.4|0.8|6.6|43.2|
|till70epoch_beam60_ctc0.3/test_clean|2620|65818|97.4|1.8|0.8|0.4|2.9|27.8|
|till70epoch_beam60_ctc0.3/test_other|2939|65101|94.3|4.0|1.6|0.8|6.4|45.3|
|till70epoch_beam60_ctc0.3_lm0.6/dev_clean|2703|68010|98.0|1.4|0.6|0.3|2.3|23.8|
|till70epoch_beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.2|3.4|1.4|0.7|5.5|36.6|
|till70epoch_beam60_ctc0.3_lm0.6/test_clean|2620|65818|97.7|1.5|0.8|0.3|2.5|24.0|
|till70epoch_beam60_ctc0.3_lm0.6/test_other|2939|65101|95.2|3.1|1.7|0.6|5.4|39.6|



# Conformer, S4 Decoder
- Params: 113.20M
- ASR config [conf/tuning/train_asr_s4_decoder.yaml](conf/tuning/train_asr_s4_decoder.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://huggingface.co/espnet/kmiyazaki_librispeech_asr_s4_decoder](https://huggingface.co/espnet/kmiyazaki_librispeech_asr_s4_decoder)
# RESULTS
## Environments
- date: `Thu Dec 29 11:58:25 UTC 2022`
- python version: `3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.0`
- Git hash: `617189d2d7e060bbcf670ab54b88776333b5137e`
  - Commit date: `Mon Dec 26 18:01:58 2022 +0900`

## asr_train_asr_s4_decoder_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|54402|98.2|1.6|0.2|0.2|2.0|25.9|
|beam60_ctc0.3/dev_other|2864|50948|95.5|4.2|0.4|0.5|5.0|42.2|
|beam60_ctc0.3/test_clean|2620|52576|98.0|1.8|0.2|0.3|2.3|27.2|
|beam60_ctc0.3/test_other|2939|52343|95.6|4.0|0.4|0.6|5.0|44.4|
|beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.5|1.3|0.2|0.2|1.7|23.0|
|beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.4|3.3|0.3|0.4|4.0|36.6|
|beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.3|1.5|0.2|0.2|1.9|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|52343|96.3|3.3|0.4|0.4|4.1|39.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.6|25.9|
|beam60_ctc0.3/dev_other|2864|265951|98.4|1.0|0.6|0.5|2.1|42.2|
|beam60_ctc0.3/test_clean|2620|281530|99.5|0.3|0.2|0.2|0.7|27.2|
|beam60_ctc0.3/test_other|2939|272758|98.6|0.8|0.6|0.6|2.0|44.4|
|beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.6|0.2|0.2|0.2|0.6|23.0|
|beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.6|0.8|0.5|0.5|1.8|36.6|
|beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.6|0.2|0.2|0.2|0.6|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|272758|98.8|0.7|0.6|0.5|1.7|39.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|68010|97.8|1.6|0.6|0.4|2.5|25.9|
|beam60_ctc0.3/dev_other|2864|63110|94.5|4.3|1.3|0.9|6.4|42.2|
|beam60_ctc0.3/test_clean|2620|65818|97.5|1.7|0.7|0.4|2.8|27.2|
|beam60_ctc0.3/test_other|2939|65101|94.6|3.9|1.5|0.8|6.2|44.4|
|beam60_ctc0.3_lm0.6/dev_clean|2703|68010|98.1|1.4|0.5|0.4|2.2|23.0|
|beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.4|3.5|1.1|0.9|5.5|36.6|
|beam60_ctc0.3_lm0.6/test_clean|2620|65818|98.0|1.4|0.6|0.4|2.4|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|65101|95.5|3.2|1.3|0.8|5.4|39.5|



# Conformer, `hop_length=160`
- Params: 116.15 M
- ASR config: [conf/tuning/train_asr_conformer10_hop_length160.yaml](conf/tuning/train_asr_conformer10_hop_length160.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://huggingface.co/pyf98/librispeech_conformer_hop_length160](https://huggingface.co/pyf98/librispeech_conformer_hop_length160)

# RESULTS
## Environments
- date: `Mon Mar 14 12:26:10 EDT 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `467660021998c416ac366aed0f75f3399e321a3a`
  - Commit date: `Sun Mar 13 17:08:56 2022 -0400`

## asr_train_asr_conformer10_hop_length160_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|54402|98.1|1.7|0.2|0.2|2.1|27.7|
|beam60_ctc0.3/dev_other|2864|50948|95.3|4.3|0.4|0.5|5.2|44.1|
|beam60_ctc0.3/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|27.9|
|beam60_ctc0.3/test_other|2939|52343|95.4|4.1|0.4|0.6|5.2|44.8|
|beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.4|1.4|0.2|0.2|1.8|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.4|3.2|0.4|0.4|3.9|36.2|
|beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.3|1.5|0.2|0.2|2.0|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|52343|96.2|3.3|0.4|0.5|4.2|39.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|27.7|
|beam60_ctc0.3/dev_other|2864|265951|98.4|1.0|0.6|0.6|2.2|44.1|
|beam60_ctc0.3/test_clean|2620|281530|99.4|0.3|0.3|0.2|0.8|27.9|
|beam60_ctc0.3/test_other|2939|272758|98.5|0.9|0.7|0.6|2.1|44.8|
|beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.5|0.2|0.2|0.2|0.6|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.5|0.8|0.6|0.5|1.9|36.2|
|beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.5|0.2|0.3|0.2|0.7|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|272758|98.6|0.7|0.7|0.5|1.9|39.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|68010|97.6|1.7|0.6|0.4|2.7|27.7|
|beam60_ctc0.3/dev_other|2864|63110|94.2|4.3|1.5|0.9|6.7|44.1|
|beam60_ctc0.3/test_clean|2620|65818|97.4|1.8|0.8|0.4|3.0|27.9|
|beam60_ctc0.3/test_other|2939|65101|94.4|3.9|1.7|0.8|6.4|44.8|
|beam60_ctc0.3_lm0.6/dev_clean|2703|68010|98.0|1.4|0.6|0.3|2.3|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.2|3.4|1.4|0.6|5.5|36.2|
|beam60_ctc0.3_lm0.6/test_clean|2620|65818|97.8|1.4|0.8|0.3|2.5|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|65101|95.1|3.2|1.7|0.6|5.5|39.6|



# Conformer, using stochastic depth
- Params: 116.15M
- ASR config [conf/tuning/train_asr_conformer9_layerdrop0.1_last6.yaml](conf/tuning/train_asr_conformer9_layerdrop0.1_last6.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://huggingface.co/pyf98/librispeech_conformer_layerdrop0.1_last6](https://huggingface.co/pyf98/librispeech_conformer_layerdrop0.1_last6)

# RESULTS
## Environments
- date: `Mon Mar  7 12:21:40 EST 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `c3569453a408fd4ff4173d9c1d2062c88d1fc060`
  - Commit date: `Sun Mar 6 23:58:36 2022 -0500`

## asr_train_asr_conformer9_layerdrop0.1_last6_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|54402|98.1|1.8|0.2|0.2|2.1|26.6|
|beam60_ctc0.3/dev_other|2864|50948|95.4|4.2|0.4|0.5|5.1|43.3|
|beam60_ctc0.3/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.4|28.1|
|beam60_ctc0.3/test_other|2939|52343|95.3|4.3|0.4|0.7|5.4|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.4|1.4|0.2|0.2|1.8|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.4|3.2|0.4|0.4|4.0|36.5|
|beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.2|1.6|0.2|0.2|2.0|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|52343|96.2|3.4|0.5|0.5|4.3|40.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|26.6|
|beam60_ctc0.3/dev_other|2864|265951|98.3|1.0|0.7|0.6|2.3|43.3|
|beam60_ctc0.3/test_clean|2620|281530|99.5|0.3|0.3|0.2|0.8|28.1|
|beam60_ctc0.3/test_other|2939|272758|98.4|1.0|0.7|0.6|2.3|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.5|0.3|0.3|0.2|0.7|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.5|0.8|0.7|0.5|1.9|36.5|
|beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.5|0.2|0.3|0.2|0.7|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|272758|98.5|0.7|0.7|0.5|2.0|40.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|68010|97.6|1.7|0.7|0.3|2.7|26.6|
|beam60_ctc0.3/dev_other|2864|63110|94.2|4.3|1.5|0.8|6.6|43.3|
|beam60_ctc0.3/test_clean|2620|65818|97.4|1.8|0.8|0.3|2.9|28.1|
|beam60_ctc0.3/test_other|2939|65101|94.2|4.1|1.7|0.8|6.6|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|68010|97.9|1.4|0.7|0.3|2.4|23.3|
|beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.2|3.4|1.5|0.6|5.5|36.5|
|beam60_ctc0.3_lm0.6/test_clean|2620|65818|97.7|1.5|0.8|0.3|2.6|23.7|
|beam60_ctc0.3_lm0.6/test_other|2939|65101|95.0|3.2|1.8|0.6|5.6|40.4|



# Conformer, new SpecAug, using weight decay in Adam
- Params: 116.15M
- ASR config [conf/tuning/train_asr_conformer8.yaml](conf/tuning/train_asr_conformer8.yaml)
- LM config: [conf/tuning/train_lm_transformer2.yaml](conf/tuning/train_lm_transformer2.yaml)
- Pretrained model: [https://huggingface.co/pyf98/librispeech_conformer](https://huggingface.co/pyf98/librispeech_conformer)

# RESULTS
## Environments
- date: `Mon Mar  7 12:26:10 EST 2022`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `c3569453a408fd4ff4173d9c1d2062c88d1fc060`
  - Commit date: `Sun Mar 6 23:58:36 2022 -0500`

## asr_train_asr_conformer8_raw_en_bpe5000_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|54402|98.1|1.8|0.2|0.2|2.1|27.3|
|beam60_ctc0.3/dev_other|2864|50948|95.2|4.4|0.4|0.5|5.4|43.7|
|beam60_ctc0.3/test_clean|2620|52576|97.9|1.9|0.2|0.3|2.3|29.0|
|beam60_ctc0.3/test_other|2939|52343|95.2|4.3|0.4|0.6|5.4|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|54402|98.4|1.4|0.2|0.2|1.8|23.5|
|beam60_ctc0.3_lm0.6/dev_other|2864|50948|96.2|3.4|0.4|0.4|4.1|37.4|
|beam60_ctc0.3_lm0.6/test_clean|2620|52576|98.3|1.5|0.2|0.2|1.9|24.1|
|beam60_ctc0.3_lm0.6/test_other|2939|52343|96.2|3.3|0.5|0.5|4.3|39.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|288456|99.5|0.3|0.2|0.2|0.7|27.3|
|beam60_ctc0.3/dev_other|2864|265951|98.2|1.1|0.7|0.6|2.4|43.7|
|beam60_ctc0.3/test_clean|2620|281530|99.4|0.3|0.3|0.2|0.8|29.0|
|beam60_ctc0.3/test_other|2939|272758|98.4|0.9|0.7|0.6|2.2|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|288456|99.5|0.2|0.2|0.2|0.7|23.5|
|beam60_ctc0.3_lm0.6/dev_other|2864|265951|98.5|0.9|0.7|0.5|2.0|37.4|
|beam60_ctc0.3_lm0.6/test_clean|2620|281530|99.5|0.2|0.3|0.2|0.7|24.1|
|beam60_ctc0.3_lm0.6/test_other|2939|272758|98.6|0.7|0.7|0.5|1.9|39.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|beam60_ctc0.3/dev_clean|2703|68010|97.6|1.8|0.7|0.3|2.8|27.3|
|beam60_ctc0.3/dev_other|2864|63110|94.1|4.4|1.5|0.9|6.8|43.7|
|beam60_ctc0.3/test_clean|2620|65818|97.4|1.8|0.7|0.3|2.9|29.0|
|beam60_ctc0.3/test_other|2939|65101|94.2|4.1|1.7|0.8|6.6|45.7|
|beam60_ctc0.3_lm0.6/dev_clean|2703|68010|97.9|1.5|0.7|0.3|2.4|23.5|
|beam60_ctc0.3_lm0.6/dev_other|2864|63110|95.1|3.5|1.4|0.6|5.6|37.4|
|beam60_ctc0.3_lm0.6/test_clean|2620|65818|97.7|1.5|0.8|0.3|2.5|24.1|
|beam60_ctc0.3_lm0.6/test_other|2939|65101|95.1|3.2|1.7|0.6|5.5|39.9|



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


## CTC decoding with nbest rescoring from decoder and LM (using k2)
With configure file:
egs2/librispeech/asr1/conf/decode_asr_transformer_with_k2.yaml

### WER
Test with a single Tesla V100 gpu and batch_size==2

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|Decoding time, seconds|
|---|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/dev_clean|2703|54402|98.2|1.6|0.2|0.2|2.0|26.5|613|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/dev_other|2864|50948|95.1|4.3|0.5|0.5|5.4|42.7|959|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/test_clean|2620|52576|98.0|1.8|0.2|0.3|2.3|27.6|618|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/test_other|2939|52343|95.2|4.2|0.5|0.5|5.3|44.7|970|


### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/dev_clean|2703|288456|99.5|0.3|0.3|0.2|0.7|26.5|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/dev_other|2864|265951|98.1|1.0|0.9|0.6|2.5|42.7|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/test_clean|2620|281530|99.4|0.3|0.3|0.2|0.8|27.6|
|decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave_use_k2_is_ctc_decoding_true_use_nbest_rescoring_true/test_other|2939|272758|98.2|0.9|0.9|0.6|2.3|44.7|


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
