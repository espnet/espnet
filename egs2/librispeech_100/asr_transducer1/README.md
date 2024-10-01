# OFFLINE SYSTEMS

## Conformer/RNN Transducer (asr_train_conformer-rnn_transducer_raw_en_bpe500_sp)

- General information
  - Pretrained model: N.A
  - Training config: conf/train_conformer-rnn_transducer.yaml
  - Decoding config: conf/decode.yaml
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 37.09 Gb
  - Training time: ~ 35 hours
  - Decoding time (32 jobs, 1 thread): ~15,6 minutes w/ default beam search.

- Environments
  - date: `Fri Feb 10 09:27:45 UTC 2023`
  - python version: `3.8.10 (default, Nov 14 2022, 12:59:47)  [GCC 9.4.0]`
  - espnet version: `espnet 202301`
  - pytorch version: `pytorch 1.8.1+cu111`
  - Git hash: `01893f855ca1a3a3645547ee4d3eaf461f7601bf`
  - Commit date: `Thu Feb 9 10:04:57 2023 +0000`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.9|4.7|0.5|0.6|5.8|53.6|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|84.9|13.4|1.6|1.8|16.9|78.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|94.6|4.8|0.6|0.6|6.0|54.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|84.7|13.6|1.8|1.6|17.0|80.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.4|0.9|0.6|0.6|2.1|53.6|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.8|3.7|2.4|1.8|8.0|78.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.4|0.9|0.7|0.5|2.2|54.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.9|3.6|2.5|1.7|7.9|80.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.5|3.4|1.1|0.6|5.0|53.6|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|86.2|10.6|3.2|2.0|15.9|78.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|95.5|3.2|1.3|0.6|5.1|54.8|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|86.0|10.4|3.5|1.9|15.9|80.1|

## E-Branchformer/RNN Transducer (asr_train_ebranchformer-rnn_transducer_raw_en_bpe500_sp)

- General information
  - Pretrained model: N.A
  - Training config: conf/train_ebranchformer-rnn_transducer.yaml
  - Decoding config: conf/decode.yaml
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 37.39 Gb
  - Training time: ~ 33,8 hours
  - Decoding time (32 jobs, 1 thread): ~15,7 minutes w/ default beam search.

- Environments
  - date: Tue Feb 14 07:41:14 UTC 2023`
  - python version: `3.8.10 (default, Nov 14 2022, 12:59:47)  [GCC 9.4.0]`
  - espnet version: `espnet 202301`
  - pytorch version: `pytorch 1.8.1+cu111`
  - Git hash: `01893f855ca1a3a3645547ee4d3eaf461f7601bf`
  - Commit date: `Thu Feb 9 10:04:57 2023 +0000`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.9|4.7|0.4|0.6|5.7|53.0|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|85.0|13.4|1.6|1.8|16.8|77.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|94.6|4.9|0.5|0.6|6.0|55.5|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|84.7|13.6|1.8|1.7|17.1|80.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.5|0.9|0.6|0.6|2.1|53.0|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.9|3.7|2.4|1.8|7.9|77.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.4|0.9|0.7|0.5|2.1|55.5|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.9|3.6|2.5|1.8|7.9|80.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.6|3.3|1.1|0.6|5.0|53.0|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|86.2|10.6|3.2|2.0|15.8|77.9|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|95.4|3.2|1.3|0.6|5.1|55.5|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|85.9|10.5|3.6|2.0|16.0|80.6|

## E-Branchformer/MEGA Transducer (asr_train_ebranchformer-mega_transducer_raw_en_bpe500_sp)

- General information
  - Pretrained model: N.A
  - Training config: conf/train_ebranchformer-mega_transducer.yaml
  - Decoding config: conf/decode.yaml
  - GPU: Nvidia A100 40Gb
  - CPU: AMD EPYC 7502P 32c
  - Peak VRAM usage during training: 37.39 Gb
  - Training time: ~ 48,9 hours
  - Decoding time (32 jobs, 1 thread): N.A

- Environments
  - date: Tue Jun 06 05:30:22 UTC 2023`
  - python version: `3.8.10 (default, Nov 14 2022, 12:59:47)  [GCC 9.4.0]`
  - espnet version: `espnet 202301`
  - pytorch version: `pytorch 1.8.1+cu111`
  - Git hash: `6048cbb8c93019c3931070c7ab0298a2f626945d`
  - Commit date: `Thu Feb 9 10:04:57 2023 +0000`

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.9|4.6|0.4|0.6|5.6|53.0|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|85.2|13.2|1.6|1.7|16.5|77.3|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|94.6|4.8|0.6|0.7|6.1|55.2|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|84.9|13.3|1.7|1.7|16.7|79.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.4|0.9|0.7|0.6|2.2|53.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.9|3.7|2.4|1.9|8.0|77.3|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.4|0.9|0.7|0.6|2.2|55.2|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.9|3.6|2.5|1.8|7.9|79.4|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.4|3.4|1.2|0.6|5.2|53.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|86.2|10.6|3.2|2.1|15.9|77.3|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|95.4|3.3|1.3|0.6|5.3|55.2|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|86.0|10.4|3.6|2.0|16.0|79.4|


# STREAMING SYSTEMS

## Conformer/RNN Transducer (asr_train_conformer-rnn_transducer_streaming_raw_en_bpe500_sp)

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

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|54402|94.3|5.2|0.5|0.7|6.4|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|50948|83.4|14.8|1.8|1.9|18.5|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|52576|93.8|5.6|0.7|0.8|7.0|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|52343|82.9|15.0|2.0|1.8|18.9|83.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|288456|98.2|1.0|0.8|0.6|2.4|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|265951|93.1|4.1|2.9|1.9|8.9|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|281530|98.0|1.1|0.9|0.6|2.6|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|272758|93.0|4.0|3.0|1.8|8.9|83.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_model_valid.loss.ave_10best/dev_clean|2703|107929|95.0|3.6|1.4|0.6|5.5|56.9|
|decode_asr_model_valid.loss.ave_10best/dev_other|2864|98610|84.7|11.6|3.6|2.2|17.4|82.1|
|decode_asr_model_valid.loss.ave_10best/test_clean|2620|105724|94.7|3.7|1.6|0.6|6.0|58.9|
|decode_asr_model_valid.loss.ave_10best/test_other|2939|101026|84.3|11.6|4.1|2.0|17.7|83.5|
