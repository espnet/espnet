# AphasiaBank English ASR recipe

## Environments

- date: `Sun Jan  8 19:23:29 EST 2023`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202211`
- pytorch version: `pytorch 1.8.1`
- Git hash: `39c1ec0509904f16ac36d25efc971e2a94ff781f`
    - Commit date: `Wed Dec 21 12:50:18 2022 -0500`

## asr_train_asr_ebranchformer_small_wavlm_large1

- [train_asr_ebranchformer_small_wavlm_large.yaml](conf/tuning/train_asr_ebranchformer_small_wavlm_large.yaml)
- Control group data is included
- [Hugging Face](https://huggingface.co/espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_small_wavlm_large1)

### WER

| dataset                             | Snt   | Wrd    | Corr | Sub  | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|------|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 120684 | 77.5 | 16.4 | 6.1 | 4.2 | 26.7 | 70.8  |

### CER

| dataset                             | Snt   | Wrd    | Corr | Sub | Del | Ins | Err  | S.Err |
|-------------------------------------|-------|--------|------|-----|-----|-----|------|-------|
| decode_asr_model_valid.acc.ave/test | 16380 | 530731 | 87.6 | 5.4 | 6.9 | 4.7 | 17.0 | 70.8  |

## asr_train_asr_ebranchformer_small_wavlm_large

- [train_asr_ebranchformer_small_wavlm_large.yaml](conf/tuning/train_asr_ebranchformer_small_wavlm_large.yaml)
- Control group data is included
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
