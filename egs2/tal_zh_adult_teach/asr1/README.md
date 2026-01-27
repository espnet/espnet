
# Branchformer Results

## Environments
- date: `Wed Dec 17 15:20:29 EST 2025`
- python version: `3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]`
- espnet2 version: `espnet2 202511`
- pytorch version: `pytorch 2.6.0+cu126`
- Git hash: `c61e21170895255cb93d1d3857bc15b818daca99`
  - Commit date: `Mon Dec 15 04:37:35 2025 -0800`

## Results

<!-- https://huggingface.co/espnet/xun_tal_zh_adult_teach_branchformer -->

- Model link: [https://huggingface.co/espnet/xun_tal_zh_adult_teach_branchformer](https://huggingface.co/espnet/xun_tal_zh_adult_teach_branchformer)
- ASR config: [./conf/train_asr_branchformer_e24_amp.yaml](./conf/train/train_asr_branchformer_e24_amp.yaml)

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|6072|160057|89.4|8.2|2.4|0.9|11.4|78.6|
|decode_asr_branchformer_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|6072|160057|89.2|8.2|2.7|1.1|11.9|78.5|
|decode_asr_branchformer_asr_model_valid.acc.ave/org/dev|3208|77334|91.3|7.2|1.5|0.8|9.6|72.8|
|decode_asr_branchformer_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/org/dev|3208|77334|90.9|7.4|1.7|1.0|10.1|72.5|
