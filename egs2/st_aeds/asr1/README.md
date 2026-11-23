# RESULTS

## Environments
- date: `Wed Aug 19 23:31:06 UTC 2026`
- python version: `3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu128`
- Git hash: `02267faa3af0467bcb804c61c126a5e8584c9546`
  - Commit date: `Thu Aug 6 07:59:17 2026 -0400`
- Pretrained Model: https://huggingface.co/JacobPercy/st_aeds_asr_whisper_medium_finetune

## exp/asr_asr_train_asr_whisper_medium_finetune_raw_en_whisper_multilingual

- ASR config: [conf/tuning/train_asr_whisper_medium_finetune.yaml](conf/tuning/train_asr_whisper_medium_finetune.yaml)
- Decode config: [conf/tuning/decode_asr_whisper_noctc_beam10.yaml](conf/tuning/decode_asr_whisper_noctc_beam10.yaml)
- Model: OpenAI Whisper Medium fine-tuning
- Decoding: beam10, no CTC, no LM

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|decode_asr_whisper_noctc_beam10_asr_model_valid.acc.ave/org/dev|400|3189|97.6|1.4|1.0|1.5|3.9|13.8|
|decode_asr_whisper_noctc_beam10_asr_model_valid.acc.ave/test|400|3231|98.0|1.1|0.9|0.1|2.1|11.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|decode_asr_whisper_noctc_beam10_asr_model_valid.acc.ave/org/dev|400|16161|98.9|0.2|0.8|1.6|2.7|13.8|
|decode_asr_whisper_noctc_beam10_asr_model_valid.acc.ave/test|400|16499|99.2|0.2|0.6|0.1|0.9|11.8|

## Data

- Dataset: ST-AEDS, OpenSLR SLR45
- Archive: `ST-AEDS-20180100_1-OS.tgz`
- License: Creative Commons BY-NC-ND 4.0
- Split: repeated prompts are kept in train when possible. Dev and test use
  only corpus-unique transcripts after lowercase and whitespace normalization.
- Text overlap: none across train, dev, and test after the same normalization.

|dataset|utterances|hours|
|---|---:|---:|
|train|3038|3.77|
|dev|400|0.48|
|test|400|0.49|
