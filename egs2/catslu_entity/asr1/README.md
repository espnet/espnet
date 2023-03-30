# Using XLS-R pretrained speech Encoder and mBART-50 Large pretrained text Encoder-Decoder

- ASR config: [conf/tuning/train_asr_branchformer_xlsr_mbart.yaml](conf/tuning/train_asr_branchformer_xlsr_mbart.yaml)

## Environments
- date: `Wed Nov 23 11:38:33 CET 2022`
- python version: `3.9.15 (main, Nov  9 2022, 00:00:00)  [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1+cu116`
- Git hash: `24cfde7f3fa3a8e4abab56c8238f7ab45e757507`
- Commit date: `Tue Nov 22 16:23:07 2022 +0100`
- Model link: [https://zenodo.org/record/7374635#.Y4VHNdLMJp8](https://zenodo.org/record/7374635#.Y4VHNdLMJp8)

## Test results

| Domain |  F1   | Acc.  |
|--------|-------|-------|
|Map     | 49.20 | 40.46 |
|Music   | 73.24 | 54.87 |
|Weather | 70.89 | 56.73 |
|Video   | 66.21 | 45.07 |
|Average | 64.89 | 49.28 |
