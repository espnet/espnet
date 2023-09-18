# BIBLE-TTS Results
## Environments

  - python version: `3.8.18`

  - pytorch version: `pytorch 2.0.1+cu117`

## The details of BibleTTS dataset

  | Languages  | Number of Training Utters | Hours (h) |
  |:-----: | :-----: | :----: |
  | Yoruba |  7,491  |  33.3  |
  | Lingala | 11,093  | 71.6 |
  | Asante twi |  21,348 | 74.9 |
  | Hausa | 40,215 | 86.6  |
  | Ewe |  22,192 |  86.8  |
  | Akuapem Twi | 24,892  | 67.1  |




## Results: using objective evaluation (MCD, RMSE) 
Pretrained models can be found at [BibleTTS pretrained models](https://huggingface.co/espnet/vits_tts_bibletts_char/tree/main)

  * Experiment results on different languages:
    |   Languages     | MCD | RMSE |
    |:-----: | :-----: | :----: |
    | Yoruba |  9.66 ± 1.05  | 0.32 ± 0.03   |
    | Lingala |  6.98 ± 1.22 | 0.34 ± 0.07 |
    | Asante twi | 8.33 ± 1.17 | 0.27 ± 0.04 |
    | Hausa | 9.52 ± 1.88 | 0.27 ± 0.07|
    | Ewe | 11.11 ± 3.62 |  0.27 ± 0.06 |
    | Akuapem Twi |  7.98 ± 1.57 | 0.28 ± 0.06 |

  * Experiment results (Ablation study on Yoruba): 
    |      Settings      | MCD_test |MCD_dev | RMSE_test | RMSE_dev |
    |:-----: | :-----: | :----: | :----: | :----: | 
    | Yoruba (train from scratch) | **9.66 ± 1.05** | **6.94 ± 1.96** | **0.32 ± 0.03** | **0.30 ± 0.06** |
    | Yoruba (finetune from pretrained model, char-based)) |  10.43 ± 1.13  | 7.97 ± 2.37 | 0.39 ± 0.05 | 0.39 ± 0.10|
    | Yoruba (finetune from pretrained model, phn-based) |  10.14 ± 1.26  | 7.42 ± 1.76 | 0.35 ± 0.04 | 0.31 ± 0.07 |

