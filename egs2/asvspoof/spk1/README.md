# RESULTS
## Overall results

| Model (conf name) | EER(%) | minDCF | Note | Huggingface |
|---|---|---|---|---|
| [conf/train_ECAPA_mel.yaml](conf/train_ECAPA_mel.yaml) | 26.124 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_ecapa_mel |
| [conf/train_xvector.yaml](conf/train_xvector.yaml) | 25.847 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_xvector_mel |
| [conf/train_mfa_conformer.yaml](conf/train_mfa_conformer.yaml) | 24.712	 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_mfaconformer_mel |
| [conf/train_SKA_mel.yaml](conf/train_SKA_mel.yaml) | 21.756 | 1.00000 | | https://huggingface.co/espnet/voxcelebs12_ska_mel |
| [conf/train_rawnet3.yaml](conf/train_rawnet3.yaml) | 17.413 | 0.99984 | | https://huggingface.co/espnet/voxcelebs12_rawnet3 |

## EER (%) breakdown for SV and A07-A19 spoofing

| Model (conf name) | SV | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [conf/train_ECAPA_mel.yaml](conf/train_ECAPA_mel.yaml) | 2.103 |39.276 | 27.635| 4.538 |51.974|51.669|42.737|18.566|39.907|44.488|57.530|3.154|3.762|7.184|
| [conf/train_xvector.yaml](conf/train_xvector.yaml) |  |  | |  |
| [conf/train_mfa_conformer.yaml](conf/train_mfa_conformer.yaml) | 	 |  | |  |
| [conf/train_SKA_mel.yaml](conf/train_SKA_mel.yaml) | 1.245 | 28.045 | 14.774 |2.775|49.013| 45.645 |37.654 |10.501|30.057|38.585 |55.800 | 1.546 |1.770 | 4.097 |
| [conf/train_rawnet3.yaml](conf/train_rawnet3.yaml) |1.117| 4.172|12.421 |1.913 |50.204|43.333|39.438|12.272|27.002|39.296|13.948|1.791|2.177|1.079

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
date: 2024-02-27 13:17:37.638194

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.0997 | 0.0997 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 1.245 | 0.06959 |
### A07
date: 2024-02-27 13:39:52.524862

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1156 | 0.1156 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 28.045 | 0.87993 |
### A08
date: 2024-02-27 13:47:13.719990

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1101 | 0.1101 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 14.774 | 0.58913 |
### A09
date: 2024-02-27 13:54:41.520466

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1038 | 0.1038 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 2.775 | 0.12594 |
### A10
date: 2024-02-27 14:01:47.901840

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1370 | 0.1370 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 49.013 | 0.99981 |
### A11
date: 2024-02-27 14:09:13.161921

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1331 | 0.1331 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 45.645 | 0.99814 |
### A12
date: 2024-02-27 14:16:24.912623

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1411 | 0.1411 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 37.654 | 0.99902 |
### A13
date: 2024-02-27 14:23:29.051876

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1209 | 0.1209 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 10.501 | 0.48012 |
### A14
date: 2024-02-27 14:31:26.798849

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1231 | 0.1231 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 30.057 | 0.98078 |
### A15
date: 2024-02-27 14:38:11.396463

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1186 | 0.1186 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 38.585 | 0.99590 |
### A16
date: 2024-02-27 14:45:03.819763

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1287 | 0.1287 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 55.800 | 1.00000 |
### A17
date: 2024-02-27 14:51:48.948318

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1059 | 0.1059 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 1.546 | 0.09163 |
### A18
date: 2024-02-27 14:59:00.361763

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.0976 | 0.0976 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 1.770 | 0.11161 |
### A19
date: 2024-02-27 15:05:46.297891

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7505 | 0.1179 |
| Non-target | 0.1128 | 0.1128 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_SKA_mel | 4.097 | 0.26304 |

## ECAPA-TDNN
### SV
date: 2024-02-27 15:20:39.941932

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.0938 | 0.0938 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 2.103 | 0.11360 |
### A07
date: 2024-02-27 15:24:48.794785

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1148 | 0.1148 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 39.276 | 0.97851 |
### A08
date: 2024-02-27 15:29:39.679477

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1128 | 0.1128 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 27.635 | 0.82176 |
### A09
date: 2024-02-27 15:33:45.922774

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1036 | 0.1036 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 4.538 | 0.22583 |
### A10
date: 2024-02-27 15:37:51.551945

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1435 | 0.1435 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 51.974 | 0.99851 |

### A11
date: 2024-02-27 15:42:13.396684

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1421 | 0.1421 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 51.669 | 0.99870 |
### A12
date: 2024-02-27 15:46:10.733481

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1411 | 0.1411 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 42.737 | 0.99851 |
### A13
date: 2024-02-27 15:50:06.975325

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1123 | 0.1123 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 18.566 | 0.75692 |
### A14
date: 2024-02-27 15:54:20.509230

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1155 | 0.1155 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 39.907 | 0.99405 |
### A15
date: 2024-02-27 15:58:17.130093

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1149 | 0.1149 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 44.488 | 0.98510 |
### A16
date: 2024-02-27 16:02:16.203237

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1346 | 0.1346 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 57.530 | 1.00000 |
### A17
date: 2024-02-27 16:06:21.542169

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1033 | 0.1033 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 3.154 | 0.18429 |
### A18
date: 2024-02-27 16:10:24.657875

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1007 | 0.1007 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 3.762 | 0.21424 |
### A19
date: 2024-02-27 16:14:24.949749

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8048 | 0.1338 |
| Non-target | 0.1106 | 0.1106 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ECAPA_mel | 7.184 | 0.38394 |