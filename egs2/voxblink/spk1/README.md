# Overall results
| Model (conf name) | EER(%) | minDCF | Note | Huggingface |
|---|---|---|---|---|
| [conf/train_rawnet3_vbClean.yaml](conf/train_rawnet3_vbClean.yaml) | 2.51 | 0.1858 | | https://huggingface.co/espnet/voxblinkclean_rawnet3 |
| [conf/train_rawnet3_vbFull.yaml](conf/train_rawnet3_vbFull.yaml) | 2.68 | 0.1893 | | https://huggingface.co/espnet/voxblinkfull_rawnet3 |
| [conf/train_rawnet3_voxcelebs12devs_voxblinkclean.yaml](conf/train_rawnet3_voxcelebs12devs_voxblinkclean.yaml) | 0.771 | 0.0556 | | https://huggingface.co/espnet/voxcelebs12devs_voxblinkclean_rawnet3 |
| [conf/train_rawnet3_voxcelebs12devs_voxblinkfull.yaml](conf/train_rawnet3_voxcelebs12devs_voxblinkfull.yaml) | 0.782 | 0.0655 | | https://huggingface.co/espnet/voxcelebs12devs_voxblinkfull_rawnet3 |

## Environments - train_rawnet3_voxcelebs12devs_voxblinkclean.yaml
date: 2023-12-25 20:26:32.589763

- python version: 3.9.16 (main, Mar  8 2023, 14:00:05)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 2.0.1

| | Mean | Std |
|---|---|---|
| Target | 7.7488 | 3.5893 |
| Non-target | 2.2723 | 2.2723 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3_voxcelebs12devs_voxblinkclean | 0.771 | 0.05568 |%

## Environments - train_rawnet3_voxcelebs12devs_voxblinkfull.yaml
date: 2023-12-26 11:07:43.666579

- python version: 3.9.16 (main, Mar  8 2023, 14:00:05)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 2.0.1

| | Mean | Std |
|---|---|---|
| Target | 7.7026 | 3.5975 |
| Non-target | 2.2858 | 2.2858 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3_voxcelebs12devs_voxblinkfull | 0.782 | 0.06551 |%

## Environments - train_rawnet3_vbClean.yaml
date: 2024-01-03 18:56:30.429852

- python version: 3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 1.13.1

| | Mean | Std |
|---|---|---|
| Target | 5.2187 | 4.6926 |
| Non-target | 2.5139 | 2.5139 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3_vbClean | 2.516 | 0.18585 |%

## Environments - train_rawnet3_vbFull.yaml
date: 2024-01-01 15:33:11.809307

- python version: 3.9.16 (main, May 15 2023, 23:46:34)  [GCC 11.2.0]
- espnet version: 202310
- pytorch version: 1.13.1

| | Mean | Std |
|---|---|---|
| Target | 5.1178 | 4.7160 |
| Non-target | 2.5179 | 2.5179 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_rawnet3_vbFull | 2.680 | 0.18937 |%
