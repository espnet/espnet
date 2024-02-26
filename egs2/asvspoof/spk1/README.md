# RESULTS
## Overall results
# RESULTS

Overall results
| Model (conf name) | EER(%) | minDCF | Note | Huggingface |
|---|---|---|---|---|
| [conf/train_ECAPA_mel.yaml](conf/train_ECAPA_mel.yaml) | 26.124 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_ecapa_mel |
| [conf/train_xvector.yaml](conf/train_xvector.yaml) | 25.847 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_xvector_mel |
| [conf/train_mfa_conformer.yaml](conf/train_mfa_conformer.yaml) | 24.712	 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_mfaconformer_mel |
| [conf/train_SKA_mel.yaml](conf/train_SKA_mel.yaml) | 21.756 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_ska_mel |
| [conf/train_rawnet3.yaml](conf/train_rawnet3.yaml) | 17.413 | 0.99984 | | https://huggingface.co/espnet/voxcelebs12_rawnet3 |

## Environments

### RawNet3
date: 2024-02-26 06:31:07.532219

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.2196 | 0.2196 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 17.413 | 0.99984 |

### SKA-TDNN
date: 2024-02-26 07:38:12.447927

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.2452 | 0.2452 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 21.756 | 1.00000 |

### ECAPA-TDNN
date: 2024-02-26 08:10:47.226164

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.2440 | 0.2440 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 26.124 | 1.00000 |

### MFA-Conformer
date: 2024-02-26 08:51:57.709168

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.2434 | 0.2434 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 24.712 | 1.00000 |

### x-vector
date: 2024-02-26 09:24:59.144113

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.2374 | 0.2374 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 25.847 | 1.00000 |