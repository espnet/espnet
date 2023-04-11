# E-Branchformer, 12 encoder layers

## Environments
- date: `Fri Dec 16 07:07:31 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1`
- Git hash: `26f432bc859e5e40cac1a86042d498ba7baffbb0`
  - Commit date: `Fri Dec 9 02:16:01 2022 +0000`

## asr_train_asr_e_branchformer_size256_mlp1024_e12_mactrue_raw_en_bpe500_sp

- Config: [conf/tuning/train_asr_e_branchformer_size256_mlp1024_e12_mactrue.yaml](conf/tuning/train_asr_e_branchformer_size256_mlp1024_e12_mactrue.yaml)
- Params: 35.01 M
- Model: [https://huggingface.co/pyf98/tedlium2_e_branchformer](https://huggingface.co/pyf98/tedlium2_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|14671|93.6|4.0|2.3|0.9|7.3|71.2|
|decode_asr_asr_model_valid.acc.ave/test|1155|27500|93.8|3.9|2.3|0.9|7.1|62.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|78259|97.2|0.8|2.0|0.9|3.7|71.2|
|decode_asr_asr_model_valid.acc.ave/test|1155|145066|97.2|0.8|2.0|0.9|3.7|62.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|28296|95.0|2.7|2.2|0.8|5.8|71.2|
|decode_asr_asr_model_valid.acc.ave/test|1155|52113|95.3|2.5|2.2|0.9|5.6|62.2|


# Conformer, 15 encoder layers

## Environments
- date: `Sat Dec 17 04:27:41 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1`
- Git hash: `26f432bc859e5e40cac1a86042d498ba7baffbb0`
  - Commit date: `Fri Dec 9 02:16:01 2022 +0000`

## asr_train_asr_conformer_e15_raw_en_bpe500_sp

- Config: [conf/tuning/train_asr_conformer_e15.yaml](conf/tuning/train_asr_conformer_e15.yaml)
- Params: 35.53 M
- Model: [https://huggingface.co/pyf98/tedlium2_conformer_e15](https://huggingface.co/pyf98/tedlium2_conformer_e15)

## Without LM

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|14671|93.5|4.1|2.5|1.0|7.5|70.0|
|decode_asr_asr_model_valid.acc.ave/test|1155|27500|93.4|4.0|2.6|1.0|7.6|64.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|78259|97.0|0.8|2.1|0.8|3.8|70.0|
|decode_asr_asr_model_valid.acc.ave/test|1155|145066|97.0|0.9|2.2|0.9|4.0|64.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|28296|95.0|2.8|2.2|0.8|5.9|70.0|
|decode_asr_asr_model_valid.acc.ave/test|1155|52113|95.1|2.5|2.4|0.9|5.8|64.2|


# Conformer, 12 encoder layers

## Environments
- date: `Fri Dec 16 05:04:30 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1`
- Git hash: `26f432bc859e5e40cac1a86042d498ba7baffbb0`
  - Commit date: `Fri Dec 9 02:16:01 2022 +0000`

## asr_train_asr_conformer_raw_en_bpe500_sp

- Config: [conf/tuning/train_asr_conformer.yaml](conf/tuning/train_asr_conformer.yaml)
- Params: 30.76 M
- Model: [https://huggingface.co/pyf98/tedlium2_conformer](https://huggingface.co/pyf98/tedlium2_conformer)

## Without LM

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|14671|93.1|4.4|2.5|1.0|7.8|69.7|
|decode_asr_asr_model_valid.acc.ave/test|1155|27500|93.4|4.0|2.6|1.0|7.6|64.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|78259|97.0|0.9|2.2|0.9|3.9|69.7|
|decode_asr_asr_model_valid.acc.ave/test|1155|145066|96.9|0.9|2.2|0.9|4.0|64.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/dev|466|28296|94.7|2.9|2.4|0.9|6.3|69.7|
|decode_asr_asr_model_valid.acc.ave/test|1155|52113|95.0|2.6|2.5|0.9|5.9|64.2|


# E-Branchformer with Transducer, 12 layers
## Environments
- date: `Thu Feb  9 01:29:33 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `478ba004e114e7862b05fb01112de7f7e1da3996`
  - Commit date: `Tue Feb 7 00:50:49 2023 +0000`

## asr_train_asr_transducer_e_branchformer_e12_raw_en_bpe500_sp
- ASR config: [conf/tuning/train_asr_transducer_e_branchformer_e12.yaml](conf/tuning/train_asr_transducer_e_branchformer_e12.yaml)
- Params: 26.26M
- Model link: [https://huggingface.co/pyf98/tedlium2_transducer_e_branchformer](https://huggingface.co/pyf98/tedlium2_transducer_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|14671|93.4|4.3|2.3|1.0|7.6|71.7|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|27500|93.6|4.0|2.4|1.0|7.4|63.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|78259|97.1|0.9|2.0|0.9|3.8|71.7|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|145066|97.1|0.9|2.1|0.9|3.9|63.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|28296|94.7|3.1|2.3|0.8|6.2|71.7|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|52113|95.1|2.6|2.2|0.9|5.8|63.5|


# Conformer with Transducer, 12 layers, 2048 linear units
## Environments
- date: `Wed Feb  8 22:07:40 CST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.13.1`
- Git hash: `478ba004e114e7862b05fb01112de7f7e1da3996`
  - Commit date: `Tue Feb 7 00:50:49 2023 +0000`

## asr_train_asr_transducer_conformer_e12_linear2048_raw_en_bpe500_sp
- ASR config: [conf/tuning/train_asr_transducer_conformer_e12_linear2048.yaml](conf/tuning/train_asr_transducer_conformer_e12_linear2048.yaml)
- Params: 34.62M
- Model link: [https://huggingface.co/pyf98/tedlium2_transducer_conformer_e12_linear2048](https://huggingface.co/pyf98/tedlium2_transducer_conformer_e12_linear2048)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|14671|93.3|4.5|2.3|1.1|7.8|71.2|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|27500|93.2|4.2|2.6|1.0|7.8|65.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|78259|97.0|0.9|2.1|1.0|3.9|71.2|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|145066|96.9|0.9|2.2|0.9|4.0|65.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transducer_asr_model_valid.loss.ave/dev|466|28296|94.6|3.0|2.4|0.9|6.3|71.2|
|decode_asr_transducer_asr_model_valid.loss.ave/test|1155|52113|94.8|2.7|2.5|0.9|6.0|65.6|



# E-Branchformer with CTC, 12 layers
## Environments
- date: `Fri Dec 30 20:15:46 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `e62de171f1d11015cb856f83780c61bd5ca7fa8f`
  - Commit date: `Thu Dec 29 14:18:44 2022 -0500`

## asr_train_asr_ctc_e_branchformer_e12_mlp1024_linear1024_raw_en_bpe500_sp
- ASR config: [conf/tuning/train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml](conf/tuning/train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml)
- Params: 25.28M
- Model link: [https://huggingface.co/pyf98/tedlium2_ctc_e_branchformer](https://huggingface.co/pyf98/tedlium2_ctc_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|14671|92.5|5.5|2.0|1.2|8.7|77.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|27500|92.7|4.9|2.3|1.1|8.3|70.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|78259|97.2|0.9|1.9|1.2|4.0|77.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|145066|97.1|0.9|2.0|1.1|4.0|70.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|28296|94.7|3.1|2.2|1.2|6.5|77.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|52113|95.0|2.7|2.2|1.1|6.1|70.6|



# Conformer with CTC, 12 layers, 2048 linear units
## Environments
- date: `Fri Dec 30 14:56:03 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `e62de171f1d11015cb856f83780c61bd5ca7fa8f`
  - Commit date: `Thu Dec 29 14:18:44 2022 -0500`

## asr_train_asr_ctc_conformer_e12_linear2048_raw_en_bpe500_sp
- ASR config: [conf/tuning/train_asr_ctc_conformer_e12_linear2048.yaml](conf/tuning/train_asr_ctc_conformer_e12_linear2048.yaml)
- Params: 33.64M
- Model link: [https://huggingface.co/pyf98/tedlium2_ctc_conformer_e12_linear2048](https://huggingface.co/pyf98/tedlium2_ctc_conformer_e12_linear2048)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|14671|92.4|5.4|2.2|1.2|8.9|75.1|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|27500|92.6|5.0|2.5|1.1|8.5|70.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|78259|97.0|0.9|2.1|1.2|4.2|75.1|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|145066|97.0|0.9|2.1|1.2|4.2|70.3|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|28296|94.6|3.1|2.4|1.2|6.6|75.1|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|52113|94.9|2.7|2.4|1.2|6.3|70.3|



# Conformer with CTC, 15 layers, 1024 linear units
## Environments
- date: `Fri Dec 30 08:37:09 CST 2022`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.12.1`
- Git hash: `e62de171f1d11015cb856f83780c61bd5ca7fa8f`
  - Commit date: `Thu Dec 29 14:18:44 2022 -0500`

## asr_train_asr_ctc_conformer_e15_linear1024_raw_en_bpe500_sp
- ASR config: [conf/tuning/train_asr_ctc_conformer_e15_linear1024.yaml](conf/tuning/train_asr_ctc_conformer_e15_linear1024.yaml)
- Params: 25.80M
- Model link: [https://huggingface.co/pyf98/tedlium2_ctc_conformer_e15_linear1024](https://huggingface.co/pyf98/tedlium2_ctc_conformer_e15_linear1024)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|14671|92.2|5.6|2.2|1.2|9.1|75.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|27500|92.1|5.4|2.5|1.1|9.0|72.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|78259|97.0|0.9|2.1|1.2|4.2|75.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|145066|96.9|0.9|2.2|1.2|4.3|72.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/dev|466|28296|94.5|3.1|2.4|1.2|6.7|75.3|
|decode_asr_ctc_asr_model_valid.cer_ctc.ave/test|1155|52113|94.6|2.9|2.5|1.2|6.5|72.8|



# Streaming Transformer

## Environments
- date: `Thu Nov 11 09:45:45 CST 2021`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.9.8`
- pytorch version: `pytorch 1.5.1`
- Git hash: `456e6517a47ef71d1b569cfa38b107538d9ef581`
  - Commit date: `Thu Aug 19 00:48:13 2021 +0800`

## asr_train_asr_streaming_transformer_raw_en_bpe500_sp
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|14671|90.3|7.5|2.3|1.7|11.4|82.4|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|27500|90.6|6.7|2.8|1.4|10.8|77.5|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.best/test|1155|27500|89.1|7.6|3.3|1.6|12.4|80.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|78259|96.1|1.6|2.3|1.5|5.4|82.4|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|145066|95.9|1.5|2.6|1.2|5.3|77.5|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.best/test|1155|145066|95.1|1.7|3.2|1.4|6.3|80.8|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/dev|466|27927|92.2|5.1|2.7|1.8|9.6|82.4|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.ave/test|1155|51430|92.4|4.6|3.0|1.3|8.9|77.5|
|decode_asr_streaming_lm_lm_train_lm_en_bpe500_valid.loss.ave_asr_model_valid.acc.best/test|1155|51430|91.0|5.3|3.7|1.5|10.5|80.8|
