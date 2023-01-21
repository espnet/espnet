# RESULTS

## Self-supervised learning features [HuBERT_large_ll60k, Conformer, utt_mvn](conf/tuning/train_asr_conformer_s3prlfrontend_hubert.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer.yaml)

### Environments
- date: `Mon Aug  2 19:37:05 EDT 2021`
- python version: `3.7.10 (default, Feb 26 2021, 18:47:35)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.8.0`
- Git hash: `05a7d399f37a54659f42739c859d5b85cad9cdc6`
  - Commit date: `Mon Aug 2 17:49:38 2021 +0900`
- Pretrained model: https://zenodo.org/record/5156171

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|97.2|2.5|0.3|0.4|3.1|33.2|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|98.4|1.6|0.1|0.2|1.8|22.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|99.0|0.3|0.7|0.2|1.2|43.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|99.4|0.2|0.4|0.1|0.7|35.4|



## Self-supervised learning features [Wav2Vec2_large_ll60k, Conformer, utt_mvn](conf/tuning/train_asr_conformer_s3prlfrontend_wav2vec2.yaml) with [Transformer-LM](conf/tuning/train_lm_transformer.yaml)

### Environments
- date: `Mon Aug  2 19:37:05 EDT 2021`
- python version: `3.7.10 (default, Feb 26 2021, 18:47:35)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.8.0`
- Git hash: `05a7d399f37a54659f42739c859d5b85cad9cdc6`
  - Commit date: `Mon Aug 2 17:49:38 2021 +0900`
- pretrained model: https://zenodo.org/record/5156153

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|97.4|2.3|0.3|0.2|2.8|32.0|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|98.4|1.6|0.1|0.1|1.8|19.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|99.1|0.4|0.6|0.2|1.1|42.5|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|99.3|0.3|0.4|0.1|0.8|32.4|



## Mask-CTC

- Training config: [conf/tuning/train_asr_transformer_maskctc.yaml](conf/tuning/train_asr_transformer_maskctc.yaml)
- Inference config:  [conf/tuning/inference_asr_maskctc.yaml](conf/tuning/inference_asr_maskctc.yaml)
- Pretrained model: https://huggingface.co/espnet/YosukeHiguchi_espnet2_wsj_asr_transformer_maskctc

### Environments

- date: `Wed Mar 23 04:54:11 JST 2022`
- python version: `3.8.12 (default, Oct 12 2021, 13:49:34)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.10.1`
- Git hash: `f29fc9d34f98635bca9e9f7860f3f6cb04300146`
  - Commit date: `Tue Mar 22 05:48:17 2022 +0900`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_asr_maskctc_asr_model_valid.cer_ctc.ave_10best/test_dev93|503|8234|87.2|11.6|1.2|1.0|13.9|79.3|
|inference_asr_maskctc_asr_model_valid.cer_ctc.ave_10best/test_eval92|333|5643|90.1|9.2|0.7|1.1|11.0|71.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_asr_maskctc_asr_model_valid.cer_ctc.ave_10best/test_dev93|503|48634|96.7|1.7|1.6|1.0|4.2|81.3|
|inference_asr_maskctc_asr_model_valid.cer_ctc.ave_10best/test_eval92|333|33341|97.7|1.3|1.1|1.0|3.3|76.0|



## E-Branchformer
- ASR config: [conf/tuning/train_asr_e_branchformer_e12_mlp1024_linear1024.yaml](conf/tuning/train_asr_e_branchformer_e12_mlp1024_linear1024.yaml)
- Params: 34.67M
- LM config: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/wsj_e_branchformer](https://huggingface.co/pyf98/wsj_e_branchformer)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|94.3|4.9|0.8|0.7|6.5|51.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|96.4|3.3|0.3|0.7|4.3|38.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.8|1.0|1.1|0.6|2.8|58.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.7|0.7|0.7|0.5|1.8|46.5|



## Conformer: enc=15, linear=1024
- ASR config: [conf/tuning/train_asr_conformer_e15_linear1024.yaml](conf/tuning/train_asr_conformer_e15_linear1024.yaml)
- Params: 35.20M
- LM config: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/wsj_conformer_e15_linear1024](https://huggingface.co/pyf98/wsj_conformer_e15_linear1024)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|94.2|5.1|0.8|0.7|6.5|52.5|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|96.4|3.3|0.3|0.6|4.1|37.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.8|1.0|1.2|0.6|2.8|58.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.6|0.7|0.7|0.5|1.9|46.8|



## Conformer: enc=12, linear=2048
- ASR config: [conf/tuning/train_asr_conformer_e12_linear2048.yaml](conf/tuning/train_asr_conformer_e12_linear2048.yaml)
- Params: 43.04M
- LM config: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Model link: [https://huggingface.co/pyf98/wsj_conformer_e12_linear2048](https://huggingface.co/pyf98/wsj_conformer_e12_linear2048)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|93.9|5.3|0.8|0.8|6.8|55.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|96.5|3.2|0.3|0.5|4.0|35.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.8|1.1|1.1|0.6|2.8|61.2|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.7|0.6|0.7|0.4|1.8|45.3|



## Using Transformer LM (ASR model is same as the above): lm_weight=1.2, ctc_weight=0.3, beam_size=20

- ASR config: [conf/tuning/train_asr_transformer2.yaml](conf/tuning/train_asr_transformer2.yaml)
- LM config: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Decode config:  [conf/decode.yaml](conf/decode.yaml)
- Pretrained model: https://zenodo.org/record/4243201

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_transformer_char_batch_bins350000_accum_grad2_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|94.2|5.1|0.7|0.8|6.6|53.3|
|inference_lm_lm_train_lm_transformer_char_batch_bins350000_accum_grad2_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|96.2|3.6|0.2|0.7|4.6|41.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_transformer_char_batch_bins350000_accum_grad2_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.8|1.0|1.2|0.5|2.7|59.2|
|inference_lm_lm_train_lm_transformer_char_batch_bins350000_accum_grad2_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.6|0.6|0.7|0.4|1.8|49.8|


## Update decode config (same config as the above model): lm_weight=1.2, ctc_weight=0.3, beam_size=20

- ASR config: [conf/tuning/train_asr_transformer2.yaml](conf/tuning/train_asr_transformer2.yaml)
- LM config: [conf/tuning/train_lm_adam_layers4.yaml](conf/tuning/train_lm_adam_layers4.yaml)
- Decode config:  [conf/decode.yaml](conf/decode.yaml)
- Pretrained model: https://zenodo.org/record/4003381/

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode.conf_ctc_weight0.3_lm_weight1.2_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|93.8|5.4|0.8|0.8|7.0|55.1|
|decode.conf_ctc_weight0.3_lm_weight1.2_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|96.0|3.7|0.3|0.8|4.7|42.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode.conf_ctc_weight0.3_lm_weight1.2_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.6|1.1|1.3|0.6|3.0|61.2|
|decode.conf_ctc_weight0.3_lm_weight1.2_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.5|0.7|0.8|0.5|2.0|50.2|

## Update only RNN-LM:  [Transformer](./conf/tuning/train_asr_transformer2.yaml) with [Char-LM](./conf/tuning/train_lm_adam_layers4.yaml)
### Environments
- date: `Mon Aug 24 11:52:54 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `e7d278ade57d8fba9b4f709150c4c499c75f53de`
  - Commit date: `Mon Aug 24 09:45:54 2020 +0900`
- Pretrained model: https://zenodo.org/record/4003381/

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|93.3|5.9|0.8|0.8|7.5|58.1|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|95.8|4.0|0.3|0.7|5.0|44.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.6|1.1|1.3|0.6|3.0|62.8|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.5|0.7|0.8|0.5|2.0|53.2|


## FBANK without pitch, [Transformer, bs=32, accum_grad=8, warmup_steps=30000, 100epoch](./conf/tuning/train_asr_transformer2.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### Environments
- date: `Mon Mar 23 18:20:34 JST 2020`
- python version: `3.7.5 (default, Oct 25 2019, 15:51:11)  [GCC 7.3.0]`
- espnet version: `espnet 0.7.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `8f3a0ff172dac2cb887878cda42a918737df8b91`
  - Commit date: `Wed Mar 18 10:41:54 2020 +0900`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|503|8234|92.2|6.9|0.9|1.3|9.1|62.2|
|decode_test_eval92_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|333|5643|95.1|4.6|0.3|0.8|5.7|49.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|503|48634|97.0|1.3|1.6|0.8|3.7|67.4|
|decode_test_eval92_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|333|33341|98.2|0.8|0.9|0.6|2.3|56.5|


## FBANK without pitch, [Transformer, bs=32, accum_grad=8, warmup_steps=60000, 200epoch](./conf/tuning/train_asr_transformer.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.ave|503|8234|92.2|6.9|0.9|1.1|8.9|63.2|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.ave|333|5643|94.3|5.3|0.4|1.0|6.7|54.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.ave|503|48634|97.1|1.4|1.5|0.7|3.6|67.6|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.ave|333|33341|98.1|1.0|1.0|0.7|2.6|61.6|


## FBANK without pitch, [VGG-BLSTMP](./conf/tuning/train_asr_rnn.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.best|503|8234|90.9|8.0|1.1|1.5|10.6|66.8|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.best|333|5643|94.1|5.3|0.6|1.0|6.9|54.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.best|503|48634|96.5|1.8|1.7|0.9|4.4|69.8|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.best|333|33341|97.8|1.1|1.0|0.7|2.9|62.2|
