# Streaming Conformer-RNN Transducer

- General information
  - Pretrained model: N.A
  - Training config: conf/train_conformer-rnn_transducer.streaming.yaml
  - Decoding config: conf/decode.yaml
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 36.7Gb
  - Training time: ~ 26 hours
  - Decoding time (32 jobs, 1 thread): ~9,1 minutes

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
