# Dataset
* MInDS-14: https://aclanthology.org/2021.emnlp-main.591/

### Supported Languages
* format: `{lang}-{LOCALE}`
* languages
   * `cs-CZ`
   * `de-DE`
   * `en-AU`
   * `en-GB`
   * `en-US`
   * `es-ES`
   * `fr-FR`
   * `it-IT`
   * `ko-KR`
   * `nl-NL`
   * `pl-PL`
   * `pt-PT`
   * `ru-RU`
   * `zh-CN`
---

# How to run
* In `run.sh`, edit the value of `language` to `{lang}-{LOCALE}` (ex. `es-ES`, `en-AU`)
---

tags:
- espnet
- audio
- automatic-speech-recognition
language: es-ES
license: mit
---

# RESULTS
* The results are for `es-ES`
## Environments
- date: `Mon Mar 14 22:28:37 UTC 2022`
- python version: `3.8.12 | packaged by conda-forge | (default, Jan 30 2022, 23:42:07)  [GCC 9.4.0]`
- espnet version: `espnet 0.10.7a1`
- pytorch version: `pytorch 1.10.1`
- Git hash: `d5322b2dc4844dce1d14268b6848607e2a3dee21`
  - Commit date: `Mon Mar 14 20:21:16 2022 +0000`

## asr_train_asr_raw_word
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_asr_model_valid.acc.ave_5best/test|49|4134|64.6|23.5|11.8|16.4|51.8|98.0|
|inference_asr_model_valid.acc.ave_5best/valid|47|4178|66.8|20.2|13.0|19.2|52.5|100.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_asr_model_valid.acc.ave_5best/test|49|8690|73.2|18.0|8.8|12.9|39.7|98.0|
|inference_asr_model_valid.acc.ave_5best/valid|47|8751|74.3|15.7|10.0|15.6|41.3|100.0|
___

# HuggingFace
* The trained model can be found at: https://huggingface.co/espnet/MInDS-14_es-ES
