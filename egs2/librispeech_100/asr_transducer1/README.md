# Streaming Conformer-RNN Transducer
# asr_train_conformer-rnn_transducer_streaming_raw_en_bpe500_sp

- General information
  - Pretrained model: N.A
  - Training config: conf/train_conformer-rnn_transducer_streaming.yaml
  - Decoding config: conf/decode.yaml (or conf/decode_streaming.yaml)
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 36.7Gb
  - Training time: ~ 26 hours
  - Decoding time (32 jobs, 1 thread): ~9,1 minutes (full context)

- Environments
  - date: `Fri Oct  7 12:02:29 UTC 2022`
  - python version: `3.8.10 (default, Jun 22 2022, 20:18:18)  [GCC 9.4.0]`
  - espnet version: `espnet 202209`
  - pytorch version: `pytorch 1.8.1+cu111`
  - Git hash: `2db74a9587a32b659cf4e1abb6b611d9f9551e09`
  - Commit date: `Thu Oct 6 15:01:23 2022 +0000`

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.3|5.2|0.5|0.7|6.4|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|83.4|14.8|1.8|1.9|18.5|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|93.8|5.6|0.7|0.8|7.0|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|82.9|15.0|2.0|1.8|18.9|83.5|

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.2|1.0|0.8|0.6|2.4|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.1|4.1|2.9|1.9|8.9|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.0|1.1|0.9|0.6|2.6|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.0|4.0|3.0|1.8|8.9|83.5|

## TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.0|3.6|1.4|0.6|5.5|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|84.7|11.6|3.6|2.2|17.4|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|94.7|3.7|1.6|0.6|6.0|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|84.3|11.6|4.1|2.0|17.7|83.5|

# Conformer-RNN Transducer
# asr_train_conformer-rnn_transducer_raw_en_bpe500_sp

- General information
  - Pretrained model: N.A
  - Training config: conf/train_conformer-rnn_transducer.yaml
  - Decoding config: conf/decode.yaml
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 36.4Gb
  - Training time: ~ 26 hours
  - Decoding time (32 jobs, 1 thread): ~9 minutes

- Environments
  - date: `Fri Oct  7 12:02:29 UTC 2022`
  - python version: `3.8.10 (default, Jun 22 2022, 20:18:18)  [GCC 9.4.0]`
  - espnet version: `espnet 202209`
  - pytorch version: `pytorch 1.8.1+cu111`
  - Git hash: `2db74a9587a32b659cf4e1abb6b611d9f9551e09`
  - Commit date: `Thu Oct 6 15:01:23 2022 +0000`

## WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.7|4.8|0.4|0.6|5.9|55.1|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|84.2|14.1|1.7|1.8|17.6|80.2|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|94.3|5.2|0.6|0.7|6.4|56.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|83.9|14.2|1.9|1.8|17.9|81.5|

## CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.4|0.9|0.7|0.6|2.2|55.1|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.4|3.9|2.7|1.9|8.5|80.2|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.2|1.0|0.8|0.6|2.3|56.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.5|3.8|2.7|1.8|8.3|81.5|

## TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.4|3.4|1.2|0.6|5.2|55.1|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|85.5|11.0|3.5|2.1|16.6|80.2|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|95.1|3.4|1.4|0.6|5.5|56.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|85.3|10.9|3.9|2.0|16.7|81.5|
