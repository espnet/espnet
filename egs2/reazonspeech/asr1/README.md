# Corpus

**ReazonSpeech** is an open Japanese corpus harvested from terrestrial
TV programs. It contains more than 10000h hours of Japanese speech,
sampled at 16kHz.

The dataset is available on Hugging Face. For more details, please visit:

* Dataset: https://huggingface.co/datasets/reazon-research/reazonspeech
* Paper: https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf

# RESULTS
## Environments
- date: `Sun Apr 23 13:29:04 UTC 2023`
- python version: `3.8.10 (default, Mar 13 2023, 10:26:41)  [GCC 9.4.0]`
- espnet version: `espnet 202301`
- pytorch version: `pytorch 1.12.1+cu102`
- Git hash: `9203b110547fba7609653e673097601043761ea5`
  - Commit date: `Fri Apr 21 10:15:47 2023 +0900`

## exp/asr_train_asr_conformer_raw_jp_char

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_lm_lm_train_lm_jp_char_valid.loss.ave_asr_model_valid.acc.ave/test|64|2364|87.7|3.0|9.2|2.8|15.1|46.9|

## Pretrained model

https://huggingface.co/reazon-research/reazonspeech-espnet-next
