# MVDR beamformer (mask_mse loss)
## Environments
- date: `Thu Dec  1 19:01:36 UTC 2022`
- python version: `3.7.4 (default, Aug 13 2019, 20:35:49)  [GCC 7.3.0]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.10.1+cu111`
- Git hash: `4ba0ccb6c5ee0dd6751fdd88d4d6a8f0cd61d87c`
  - Commit date: ``


## enh_train_enh_beamformer_mvdr_raw

config: ./conf/tuning/train_enh_beamformer_mvdr.yaml

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_cv_mix_single_reverb_min_8k|77.35|4.11|4.11|0.00|3.81|
|enhanced_tt_mix_single_reverb_min_8k|79.26|3.44|3.44|0.00|3.17|
