# RESULTS
## Overall results

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

# Subprotocol trials
## RawNet3
### SV
date: 2024-02-27 11:15:36.088677

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.0964 | 0.0964 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 1.117 | 0.04894 |

### A07
date: 2024-02-27 11:53:35.503210

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.0781 | 0.0781 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 4.172 | 0.14652 |
### A08
date: 2024-02-27 12:02:01.602488

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1206 | 0.1206 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 12.421 | 0.55514 |
### A09
date: 2024-02-27 12:09:55.437283

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.0966 | 0.0966 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 1.913 | 0.09700 |
### A10
date: 2024-02-27 12:15:25.247395

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1326 | 0.1326 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 50.204 | 1.00000 |
### A11
date: 2024-02-27 12:20:42.451489

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1273 | 0.1273 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 43.333 | 0.99851 |
### A12
date: 2024-02-27 12:25:56.069889

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1282 | 0.1282 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 39.438 | 0.99851 |
### A13
date: 2024-02-27 12:31:42.769969

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1019 | 0.1019 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 12.272 | 0.48230 |
### A14
date: 2024-02-27 12:37:12.244739

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1126 | 0.1126 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 27.002 | 0.97394 |
### A15
date: 2024-02-27 12:42:15.843638

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1094 | 0.1094 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 39.296 | 0.99646 |
### A16
date: 2024-02-27 12:47:57.103611

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1018 | 0.1018 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 13.948 | 0.70038 |
### A17
date: 2024-02-27 12:53:02.348708

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.1042 | 0.1042 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 1.791 | 0.08008 |
### A18
date: 2024-02-27 12:58:24.712830

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.0986 | 0.0986 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 2.177 | 0.09191 |
### A19
date: 2024-02-27 13:03:33.379022

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7689 | 0.1144 |
| Non-target | 0.0809 | 0.0809 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3 | 1.079 | 0.06626 |
## SKA-TDNN
### SV

### A07

### A08

### A09

### A10

### A11

### A12

### A13

### A14

### A15

### A16

### A17

### A18

### A19

## ECAPA-TDNN
### SV

### A07

### A08

### A09

### A10

### A11

### A12

### A13

### A14

### A15

### A16

### A17

### A18

### A19

## MFA-conformer
### SV

### A07

### A08

### A09

### A10

### A11

### A12

### A13

### A14

### A15

### A16

### A17

### A18

### A19

## x-vector
### SV

### A07

### A08

### A09

### A10

### A11

### A12

### A13

### A14

### A15

### A16

### A17

### A18

### A19