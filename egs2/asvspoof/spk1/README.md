# RESULTS
## Overall results

| Model (conf name) | SASV-EER(%) | minDCF | Note | Huggingface |
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
| [conf/train_xvector.yaml](conf/train_xvector.yaml) |1.769| 39.214 |27.432 | 2.514 |52.514|49.389|44.618|19.292|40.521|42.593|59.702|2.551|3.948|9.255|
| [conf/train_mfa_conformer.yaml](conf/train_mfa_conformer.yaml) |1.335|33.394|23.077|2.890|52.666|47.561|43.529|17.952|37.281|43.128|59.106|2.570|2.259|6.108|
| [conf/train_SKA_mel.yaml](conf/train_SKA_mel.yaml) | 1.245 | 28.045 | 14.774 |2.775|__49.013__| 45.645 |__37.654__|__10.501__|30.057|__38.585__|55.800 |__1.546__|__1.770__| 4.097 |
| [conf/train_rawnet3.yaml](conf/train_rawnet3.yaml) |__1.117__|__4.172__|__12.421__|__1.913__|50.204|__43.333__|39.438|12.272|__27.002__|39.296|__13.948__|1.791|2.177|__1.079__

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
<details><summary>expand</summary>

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

## MFA-conformer
### SV
date: 2024-02-27 17:16:47.826700

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.0938 | 0.0938 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 1.335 | 0.07360 |
### A07
date: 2024-02-27 17:22:35.591538

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1142 | 0.1142 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 33.394 | 0.91796 |
### A08
date: 2024-02-27 17:28:21.413240

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1161 | 0.1161 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 23.077 | 0.72508 |
### A09
date: 2024-02-27 17:34:06.180903

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.0992 | 0.0992 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 2.890 | 0.12334 |
### A10
date: 2024-02-27 17:39:53.286866

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1399 | 0.1399 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 52.666 | 1.00000 |
### A11
date: 2024-02-27 17:45:38.874941

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1357 | 0.1357 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 47.561 | 0.99832 |
### A12
date: 2024-02-27 17:51:26.157545

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1374 | 0.1374 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 43.529 | 0.99926 |
### A13
date: 2024-02-27 17:57:13.055793

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1114 | 0.1114 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 17.952 | 0.64849 |
### A14
date: 2024-02-27 18:02:59.994983

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1222 | 0.1222 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 37.281 | 0.98003 |
### A15
date: 2024-02-27 18:08:46.911828

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1190 | 0.1190 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 43.128 | 0.99069 |
### A16
date: 2024-02-27 18:14:34.204601

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1328 | 0.1328 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 59.106 | 1.00000 |
### A17
date: 2024-02-27 18:20:20.830238

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1065 | 0.1065 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 2.570 | 0.16197 |
### A18
date: 2024-02-27 18:26:08.860074

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.0951 | 0.0951 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 2.259 | 0.11208 |
### A19
date: 2024-02-27 18:31:56.184388

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.7894 | 0.1262 |
| Non-target | 0.1063 | 0.1063 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_mfa_conformer | 6.108 | 0.36242 |2024-02-27T18:31:56 |

## x-vector
### SV
date: 2024-02-27 18:46:47.088042

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1007 | 0.1007 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 1.769 | 0.12031 |
### A07
date: 2024-02-27 18:50:17.579856

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1134 | 0.1134 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 39.214 | 0.98642 |
### A08
date: 2024-02-27 18:53:45.686145

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1222 | 0.1222 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 27.432 | 0.86012 |
### A09
date: 2024-02-27 18:57:15.571895

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1039 | 0.1039 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 2.514 | 0.12675 |
### A10
date: 2024-02-27 19:00:45.868605

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1428 | 0.1428 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 52.514 | 0.99981 |
### A11
date: 2024-02-27 19:04:16.249534

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1380 | 0.1380 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 49.389 | 0.99888 |
### A12
date: 2024-02-27 19:07:46.341433

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1358 | 0.1358 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 44.618 | 0.99907 |
### A13
date: 2024-02-27 19:11:16.559528

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1060 | 0.1060 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 19.292 | 0.70417 |
### A14
date: 2024-02-27 19:14:46.716614

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1117 | 0.1117 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 40.521 | 0.99814 |
### A15
date: 2024-02-27 19:18:18.388321

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1121 | 0.1121 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 42.593 | 0.99907 |
### A16
date: 2024-02-27 19:21:48.705205

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1242 | 0.1242 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 59.702 | 1.00000 |
### A17
date: 2024-02-27 19:25:19.174404

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1093 | 0.1093 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 2.551 | 0.13312 |
### A18
date: 2024-02-27 19:28:49.195360

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1000 | 0.1000 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 3.948 | 0.20679 |
### A19
date: 2024-02-27 19:32:20.156885

- python version: 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0]
- espnet version: 202402
- pytorch version: 2.0.0

| | Mean | Std |
|---|---|---|
| Target | -0.8091 | 0.1173 |
| Non-target | 0.1105 | 0.1105 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_xvector | 9.255 | 0.54227 |
