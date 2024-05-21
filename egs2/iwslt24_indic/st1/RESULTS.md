# RESULTS

## En-Hi

### Environments
- date: `Thu Apr 18 01:34:53 JST 2024`
- python version: `3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `83c179ab842987cf01642df2db372aaae260df55`
  - Commit date: `Wed Apr 17 00:28:29 2024 +0900`

### Model config

- training: [./conf/tuning/train_st_conformer.yaml](./conf/tuning/train_st_conformer.yaml)
- decoding: [./conf/tuning/decode_st_conformer.yaml](./conf/tuning/decode_st_conformer.yaml)
- model url: [https://huggingface.co/espnet/iwslt24_indic_en_hi_bpe_tc4000](https://huggingface.co/espnet/iwslt24_indic_en_hi_bpe_tc4000)

### BLEU

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_st_model_valid.acc.ave/dev.en-hi|37.1|64.8/44.9/34.2/26.2 (BP = 0.924 ratio = 0.927 hyp_len = 195297 ref_len = 210636)|

## En-Bn

### Environments
- date: `Wed Apr 17 02:51:38 JST 2024`
- python version: `3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `83c179ab842987cf01642df2db372aaae260df55`
  - Commit date: `Wed Apr 17 00:28:29 2024 +0900`

### Model config

- training: [./conf/tuning/train_st_conformer.yaml](./conf/tuning/train_st_conformer.yaml)
- decoding: [./conf/tuning/decode_st_conformer.yaml](./conf/tuning/decode_st_conformer.yaml)
- model url: [https://huggingface.co/espnet/iwslt24_indic_en_bn_bpe_tc4000](https://huggingface.co/espnet/iwslt24_indic_en_bn_bpe_tc4000)

### BLEU

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_st_model_valid.acc.ave/dev.en-bn|2.1|19.7/3.6/1.0/0.3 (BP = 1.000 ratio = 1.185 hyp_len = 46094 ref_len = 38883)|

# En-Ta

## Environments
- date: `Thu Apr 18 01:03:59 JST 2024`
- python version: `3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0]`
- espnet version: `espnet 202402`
- pytorch version: `pytorch 2.1.0`
- Git hash: `83c179ab842987cf01642df2db372aaae260df55`
  - Commit date: `Wed Apr 17 00:28:29 2024 +0900`

### Model config

- training: [./conf/tuning/train_st_conformer.yaml](./conf/tuning/train_st_conformer.yaml)
- decoding: [./conf/tuning/decode_st_conformer.yaml](./conf/tuning/decode_st_conformer.yaml)
- model url: [https://huggingface.co/espnet/iwslt24_indic_en_ta_bpe_tc4000](https://huggingface.co/espnet/iwslt24_indic_en_ta_bpe_tc4000)

### BLEU

|dataset|score|verbose_score|
|---|---|---|
|decode_st_conformer_st_model_valid.acc.ave/dev.en-ta|6.3|46.5/9.4/4.7/1.9 (BP = 0.798 ratio = 0.816 hyp_len = 66168 ref_len = 81059)|
