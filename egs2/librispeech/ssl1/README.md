## INTRODUCTION

This recipe trains a [Hubert](https://arxiv.org/pdf/2106.07447.pdf)pretrain model, using data Librispeech 960hr data, including the k-means-based pseudo label generation and mask-prediction training.

================================================

## RESULTS
Detailed information, e.g. kmeans performance, accuracies, training curves, etc, can be found in the [PR page](https://github.com/espnet/espnet/pull/4747) and the following HuggingFace repos.

### iteration 0 pretrained model:
#### Environments
- date: `Wed Jan 4 08:48:57 EST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59) [GCC 11.2.0]`
- espnet version: `202209`
- pytorch version: `pytorch 1.13.0+cu117`
- Git hash: `753f40d61813436d4e76660904d02eaed7a6649e`
  - Commit date: `Wed Jan 4 06:52:27 2023 -0600`
- SSL config: [conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it0.yaml](conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it0.yaml)
- Pretrained model: [https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw](https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw)
- Finetuning performance on [LibriLight_Limited 10 hr](https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz)
  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_asr_model_valid.loss.ave/dev_clean|2694|53635|85.8|13.6|0.6|1.2|15.5|85.6|
  |decode_asr_model_valid.loss.ave/dev_other|2864|50948|78.1|20.4|1.5|2.0|23.9|91.5|
  |decode_asr_model_valid.loss.ave/test_clean|2620|52576|85.4|13.9|0.7|1.3|15.9|84.5|
  |decode_asr_model_valid.loss.ave/test_other|2939|52343|78.0|20.6|1.5|2.1|24.1|91.6|

### iteration 1 pretrained model:
#### Environments
- date: `Wed Jan 10 01:20:10 EST 2023`
- python version: `3.9.15 (main, Nov 24 2022, 14:31:59) [GCC 11.2.0]`
- espnet version: `202209`
- pytorch version: `pytorch 1.13.0+cu117`
- Git hash: `753f40d61813436d4e76660904d02eaed7a6649e`
  - Commit date: `Wed Jan 4 06:52:27 2023 -0600`
- SSL config: [conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it1.yaml](conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it1.yaml)
- Pretrained model: [https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw](https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw)
- Finetuning performance on [LibriLight_Limited 10 hr](https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz)
  |dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
  |---|---|---|---|---|---|---|---|---|
  |decode_asr_model_valid.loss.ave/dev_clean|2694|53635|90.3|9.3|0.5|0.7|10.4|74.8|
  |decode_asr_model_valid.loss.ave/dev_other|2864|50948|83.8|15.1|1.1|1.2|17.4|83.9|
  |decode_asr_model_valid.loss.ave/test_clean|2620|52576|90.2|9.4|0.4|0.7|10.5|75.2|
  |decode_asr_model_valid.loss.ave/test_other|2939|52343|83.6|15.2|1.1|1.3|17.6|85.3|
  - Pretrained model: [https://huggingface.co/espnet/simpleoier_librilight_limited_asr_train_asr_hubert_base_10h_finetuning_raw_en_char](https://huggingface.co/espnet/simpleoier_librilight_limited_asr_train_asr_hubert_base_10h_finetuning_raw_en_char)

================================================

## HUBERT IN FAIRSEQ

The original Hubert paper, code and model can be found in:
paper: https://arxiv.org/pdf/2106.07447.pdf
code and model: https://github.com/pytorch/fairseq/tree/master/examples/hubert

## HUBERT IN TORCHAUDIO

code and results: https://github.com/pytorch/audio/tree/main/examples/hubert

================================================

## ACKNOWLEDGEMENT

We would like to thank Wei-Ning Hsu(Facebook) and Abdelrahman Mohamed(Facebook) for their work on Hubert and valuable
information/kind help of this implementation.