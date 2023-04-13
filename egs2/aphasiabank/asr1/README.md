# AphasiaBank English ASR recipe

## Data preparation

1. Download AphasiaBank from https://aphasia.talkbank.org
2. See [data.sh](local/data.sh) for instructions

## Evaluation

- [local/score_cleaned.sh](local/score_cleaned.sh) is used to calculate CER/WER per Aphasia subset.
  It doesn't require the input hypothesis file to contain language or Aph tags.
  But if the input does contain, it will automatically remove them.
- [local/score_per_severity.sh](local/score_per_severity.sh) is similar, but it calculates CER/WER per Aphasia severity.
  But if the input does contain, it will automatically remove them.
- [local/score_interctc_aux.sh](local/score_interctc_aux.sh) is used to calculate InterCTC-based Aphasia
  detection accuracy.
- [local/score_aphasia_detection.py](local/score_aphasia_detection.py) is used to calculate Aphasia
  detection accuracy from input in Kaldi text format.

## Environments

- date: `Sun Jan  8 19:23:29 EST 2023`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.8.1`
- Git hash: `39c1ec0509904f16ac36d25efc971e2a94ff781f`
    - Commit date: `Wed Dec 21 12:50:18 2022 -0500`

## asr_train_asr_ebranchformer_small_wavlm_large1

- [train_asr_ebranchformer_small_wavlm_large1.yaml](conf/tuning/train_asr_ebranchformer_small_wavlm_large1.yaml)
- Control group data is included
- Downsampling rate = 2 = 2 (WavLM) * 1 (`Conv2dSubsampling1`)
- [Hugging Face](https://huggingface.co/espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_small_wavlm_large1)

### WER

| dataset | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|---------|-------|--------|------|------|-----|-----|------|-------|
| test    | 28424 | 240039 | 81.2 | 13.3 | 5.5 | 3.5 | 22.2 | 67.8  |

### CER

| dataset | Snt   | Wrd     | Corr | Sub | Del | Ins | Err  | S.Err |
|---------|-------|---------|------|-----|-----|-----|------|-------|
| test    | 28424 | 1103375 | 89.9 | 4.1 | 5.9 | 3.7 | 13.8 | 67.8  |
| PWA     | 17936 | 591922  | 87.9 | 5.4 | 6.7 | 4.6 | 16.7 | 70.7  |
| control | 10488 | 511453  | 92.3 | 2.7 | 5.0 | 2.8 | 10.5 | 62.9  |

## asr_train_asr_ebranchformer_small_wavlm_large

- [train_asr_ebranchformer_small_wavlm_large.yaml](conf/tuning/train_asr_ebranchformer_small_wavlm_large.yaml)
- Control group data is included
- Downsampling rate = 4 = 2 (WavLM) * 2 (`Conv2dSubsampling2`)
- [Hugging Face](https://huggingface.co/espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_small_wavlm_large)

### WER

| dataset                             | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|------|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 120684 | 76.6 | 16.7 | 6.7 | 3.8 | 27.1 | 72.4  |

### CER

| dataset                             | Snt   | Wrd    | Corr | Sub | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|-----|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 530731 | 87.1 | 5.3 | 7.6 | 4.9 | 17.7 | 72.4  |

## asr_train_asr_ebranchformer_small_raw_en_char_sp

- [train_asr_ebranchformer_small.yaml](conf/tuning/train_asr_ebranchformer_small.yaml)
- Control group data is included

### WER

| dataset                             | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|------|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 120684 | 69.7 | 22.7 | 7.6 | 4.5 | 34.9 | 77.5  |

### CER

| dataset                             | Snt   | Wrd    | Corr | Sub | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|-----|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 530731 | 82.8 | 8.0 | 9.2 | 5.1 | 22.3 | 77.5  |

## asr_train_asr_conformer_hubert_ll60k_large_raw_en_char_sp

- [train_asr_conformer_hubert_ll60k_large.yaml](conf/tuning/train_asr_conformer_hubert_ll60k_large.yaml)
- Control group data is included

### WER

| dataset                             | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|------|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 120684 | 68.9 | 22.8 | 8.3 | 4.4 | 35.5 | 81.5  |

### CER

| dataset                             | Snt   | Wrd    | Corr | Sub | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|-----|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 530731 | 82.1 | 8.0 | 9.9 | 5.3 | 23.3 | 81.5  |

## asr_train_asr_conformer_raw_en_char_sp

- [train_asr_conformer.yaml](conf/tuning/train_asr_conformer.yaml)
- Control group data is included

### WER

| dataset                             | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|------|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 120684 | 68.1 | 23.6 | 8.3 | 4.5 | 36.4 | 79.9  |

### CER

| dataset                             | Snt   | Wrd    | Corr | Sub | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|-----|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 530731 | 81.7 | 8.4 | 9.9 | 5.2 | 23.5 | 79.9  |
