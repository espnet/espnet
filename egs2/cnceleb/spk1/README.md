# RESULTS
## Overall results

| Model (conf name)                                        | EER(%) | minDCF | Note | Huggingface                                      |
|----------------------------------------------------------|---|---|---|--------------------------------------------------|
| [conf/train_ecapa_tdnn.yaml](conf/train_ecapa_tdnn.yaml) | 7.762 | 0.3530 | | https://huggingface.co/espnet/cnceleb_ecapa_tdnn |
| [conf/train_resnet34.yaml](conf/train_resnet34.yaml)     | 7.119 | 0.3499 | | https://huggingface.co/espnet/cnceleb_resnet34   |
| [conf/train_resnet221.yaml](conf/train_resnet221.yaml)   | 6.003 | 0.2940 | | https://huggingface.co/espnet/cnceleb_resnet221  |

## Environments - conf/train_ecapa_tdnn.yaml
date: 2025-06-17 13:12:33.497674

- python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
- espnet version: 202402
- pytorch version: 2.3.1

| | Mean | Std |
|---|---|---|
| Target | -1.0389 | 0.1553 |
| Non-target | -1.3574 | 0.0679 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_ecapa_tdnn | 7.762 | 0.35304 |

## Environments - conf/train_resnet34.yaml
date: 2025-06-17 13:11:32.771625

- python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
- espnet version: 202402
- pytorch version: 2.3.1

| | Mean | Std |
|---|---|---|
| Target | -1.0307 | 0.1501 |
| Non-target | -1.3465 | 0.0693 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_resnet34 | 7.119 | 0.34991 |

## Environments - conf/train_resnet221.yaml
date: 2025-06-17 13:11:19.656417

- python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
- espnet version: 202402
- pytorch version: 2.3.1

| | Mean | Std |
|---|---|---|
| Target | -1.0174 | 0.1565 |
| Non-target | -1.3613 | 0.0660 |

| Model name | EER(%) | minDCF |
|---|---|---|
| conf/train_resnet221 | 6.003 | 0.29403 |
