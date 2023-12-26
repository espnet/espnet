Overall results
| Model (conf name) | EER(%) | minDCF | Note | Huggingface |
|---|---|---|---|---|
| [conf/train_rawnet3_voxblinkclean.yaml](conf/train_rawnet3_voxblinkclean.yaml) | 0. | 0. | | https://huggingface.co/espnet/voxblinkclean_rawnet3 |
| [conf/train_rawnet3_voxblinkfull.yaml](conf/train_rawnet3_voxblinkfull.yaml) | 0. | 0. | | https://huggingface.co/espnet/voxblinkfull_rawnet3 |
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
