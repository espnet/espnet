# AphasiaBank English ASR recipe

## Data preparation

1. Download AphasiaBank from https://aphasia.talkbank.org
2. See [data.sh](local/data.sh) for instructions

Data splits are stored in [a separate repository](https://github.com/tjysdsg/AphasiaBank_config).

## Experiments

- Use `run.sh` for baselines and tag-based experiments.
- Use `run_interctc.sh` for InterCTC-based experiments (also supports combination of
  tag- and InterCTC-based).

Parameters:

- `--include_control`: true to include control group data
- `--tag_insertion`: `prepend`, `append`, `both`, or `none`.

## Evaluation

- Use `run.sh --nlsyms_txt none --stage 13` to score your model.
    - It's important to set `--nlsyms_txt none` to avoid removing the Aphasia tags,
      which will be used by the scripts below.
- [local/score_cleaned.sh](local/score_cleaned.sh) is used to calculate CER/WER per
  Aphasia subset.
  It doesn't require the input hypothesis file to contain language or Aph tags.
  But if the input does contain, it will automatically remove them.
- [local/score_per_severity.sh](local/score_per_severity.sh) is similar, but it
  calculates CER/WER per Aphasia severity.
- [local/score_interctc_aux.sh](local/score_interctc_aux.sh) is used to calculate
  InterCTC-based Aphasia detection accuracy.
- [local/score_aphasia_detection.py](local/score_aphasia_detection.py) is used to
  calculate Aphasia
  detection accuracy from input in Kaldi text format.
- Calculate MACS and FLOPS of the encoder
  using [this script](https://github.com/pyf98/espnet_utils/blob/master/profile.sh)

## RESULTS (WER)

**Environments**

- date: `Fri Apr 28 00:16:04 EDT 2023`
- python version: `3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0]`
- espnet version: `espnet 202301`
- chainer version: `chainer 6.0.0`
- pytorch version: `pytorch 1.8.1`
- Git hash: `5b2f8f351712aac98aa9f8370f1d2aea1ecff685`
    - Commit date: `Fri Apr 28 00:10:24 2023 -0400`

| Model                                                                                                                                                                                                        | Patient | Control | Overall | Sentence-Level Detection Accuracy | Speaker-Level Detection Accuracy |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|---------|-----------------------------------|----------------------------------|
| [Conformer](conf/tuning/train_asr_conformer.yaml)                                                                                                                                                            | 40.3    | 35.3    | 38.1    |                                   |                                  |
| [E-Branchformer(EBF)](conf/tuning/train_asr_ebranchformer_small.yaml)                                                                                                                                        | 36.2    | 31.2    | 34.0    |                                   |                                  |
| [EBF+WavLM](conf/tuning/train_asr_ebranchformer_small_wavlm_large1.yaml) ([HuggingFace](espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_small_wavlm_large1))                                         | 26.3    | 16.9    | 22.2    |                                   |                                  |
| [EBF+WavLM+Tag-prepend](conf/tuning/train_asr_ebranchformer_small_wavlm_large1.yaml)                                                                                                                         | 26.3    | 16.9    | 22.2    | 89.3                              | 95.1                             |
| [EBF+WavLM+Tag-append](conf/tuning/train_asr_ebranchformer_small_wavlm_large1.yaml)                                                                                                                          | 26.2    | 16.9    | 22.1    | 89.2                              | 95.1                             |
| [EBF+WavLM+Tag-both](conf/tuning/train_asr_ebranchformer_small_wavlm_large1.yaml) ([HuggingFace](https://huggingface.co/espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_wavlm_aph_en_both))          | 26.3    | 16.8    | 22.1    | Front: 90.8, Back: 90.6           | Front: 95.7, Back: 95.7          |
| [EBF+WavLM+InterCTC6](conf/tuning/train_asr_ebranchformer_small_wavlm_large1_interctc6.yaml) ([HuggingFace](https://huggingface.co/espnet/jiyang_tang_aphsiabank_english_asr_ebranchformer_wavlm_interctc6)) | 26.3    | 16.9    | 22.1    | 85.2                              | 97.3                             |
| [EBF+WavLM+InterCTC3+6](conf/tuning/train_asr_ebranchformer_small_wavlm_large1_interctc3+6.yaml)                                                                                                             | 26.5    | 17.1    | 22.3    | 83.5                              | 96.7                             |
| EBF+WavLM+InterCTC9 (set `interctc_layer_idx` and `aux_ctc` to 9 in the InterCTC6 config)                                                                                                                    | 26.3    | 16.9    | 22.2    | 84.5                              | 97.3                             |
| [EBF+WavLM+InterCTC6+Tag-prepend](conf/tuning/train_asr_ebranchformer_small_wavlm_large1_interctc6.yaml)                                                                                                     | 26.3    | 16.9    | 22.1    | Tag: 89.7, InterCTC: 89.6         | Tag: 96.7, InterCTC: 96.7        |
