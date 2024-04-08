# Using XLS-R pretrained speech Encoder and mBART-50 Large pretrained text Encoder-Decoder

- ASR config: [conf/tuning/train_asr_branchformer_xlsr_mbart.yaml](conf/tuning/train_asr_branchformer_xlsr_mbart.yaml)

## Environments
- date: `Wed Nov 23 11:00:08 CET 2022`
- python version: `3.9.15 (main, Nov  9 2022, 00:00:00)  [GCC 11.3.1 20220421 (Red Hat 11.3.1-3)]`
- espnet version: `espnet 202209`
- pytorch version: `pytorch 1.12.1+cu116`
- Git hash: `24cfde7f3fa3a8e4abab56c8238f7ab45e757507`
- Commit date: `Tue Nov 22 16:23:07 2022 +0100`
- Model link: [https://zenodo.org/record/7374681#.Y4VQ8NLMJp8](https://zenodo.org/record/7374681#.Y4VQ8NLMJp8)

## Results

|                                            |  CER  | CVER  |
|--------------------------------------------|-------|-------|
| decode_asr_hf_asr_model_valid.acc.ave/dev  | 23.23 | 28.70 |
| decode_asr_hf_asr_model_valid.acc.ave/test | 24.01 | 28.02 |
