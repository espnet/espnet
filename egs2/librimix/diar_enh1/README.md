# RESULTS
Paper: [EEND-SS: Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers](https://arxiv.org/abs/2203.17068)

## Environments
- date: `Fri Mar 25 18:55:59 EDT 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1+cu102`
- Git hash: `4f0f9a2435549211ef670354d09eb45883441b2d`
  - Commit date: `Tue Mar 15 10:52:24 2022 -0400`

## Conditions
- sample_rate: 8k
- min_or_max: min
- threshold for DER calculation: 0.5
- collar torelance: 0.0 sec
- median filtering: 11 frames

## Notes
- "LMF" refers to the model using the concatenation of TCN Bottleneck features and log-mel filterbank features. *To use this option, add `frontend: default` and replace `input_size: 128` of `diar_encoder_conf:` with `input_size: 208` (frontend_dim (80) + bottleneck_dim (128)) in [conf/tuning/train_diar_enh_convtasnet_2.yaml](conf/tuning/train_diar_enh_convtasnet_2.yaml) or [conf/tuning/train_diar_enh_convtasnet_adapt.yaml](conf/tuning/train_diar_enh_convtasnet_adapt.yaml)*.
- "PP" refers to the post-processing applied to the separated audio signals using the speaker activity (diarization result) during inference. *To use this option, add `multiply_diar_result: True` in  [conf/tuning/decode_diar_enh.yaml](conf/tuning/decode_diar_enh.yaml) or [conf/tuning/decode_diar_enh_adapt.yaml](conf/tuning/decode_diar_enh_adapt.yaml)*.

### Libri2Mix (2 Speakers)

||STOI|SI_SNRi|SDRi|DER|
|---|---|---|---|---|
|EEND-SS|0.826|9.76|10.57|5.08|
|+ PP|0.826|9.83|10.67|5.08|
|+ LMF|0.831|10.62|11.13|5.17|
|+ LMF + PP|0.831|10.70|11.23|5.17|

### Libri3Mix (3 Speakers)

||STOI|SI_SNRi|SDRi|DER|
|---|---|---|---|---|
|EEND-SS|0.722|7.66|8.60|6.26|
|+ PP|0.722|7.71|8.66|6.26|
|+ LMF|0.723|8.39|8.96|6.00|
|+ LMF + PP|0.723|8.40|9.00|6.00|

### Libri2&3Mix (2 & 3 Speakers)
- Model adaptation on 2 & 3 mixed speaker dataset is conducted after pre-training on 2 speaker dataset. *Run [local/run_adapt.sh](local/run_adapt.sh) after running [run.sh](run.sh) with `num_spk=2`*.
- Pre-trained models: 
   - EEND-SS: https://huggingface.co/espnet/YushiUeda_librimix_diar_enh_2_3_spk
   - +LMF: https://huggingface.co/espnet/YushiUeda_librimix_diar_enh_2_3_spk_lmf

||STOI|SI_SNRi|SDRi|DER|
|---|---|---|---|---|
|EEND-SS|0.760|9.31|7.50|6.27|
|+ PP|0.760|9.38|7.59|6.27|
|+ LMF|0.767|8.83|9.72|6.04|
|+ LMF + PP|0.767|8.87|9.77|6.04|
