# RESULTS
## Environments
- date: `Mon Sep  5 14:55:27 EDT 2022`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.10.1`
- Git hash: `6d5236553b7fb3e653907c447bbbbb0790a013f9`
  - Commit date: `Wed Aug 31 08:17:56 2022 -0400`


## enh_train_enh_mixit_conv_tasnet_raw

config: conf/tuning/train_enh_conv_tasnet.yaml

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_cv_min_8k|91.43|14.55|13.96|24.12|13.34|
|enhanced_tt_min_8k|91.32|13.68|12.91|22.61|12.25|