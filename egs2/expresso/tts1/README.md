# EXPRESSO RECIPE

This is recipe for the multi-speaker TTS model with Expresso corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

Model: VITS

## Environments
- date: `Fri May 30 16:37:38 UTC 2025`
- python version: `3.9.22 | packaged by conda-forge | (main, Apr 14 2025, 23:35:59)  [GCC 13.3.0]`
- espnet version: `espnet 202503`
- pytorch version: `pytorch 2.4.1+cu124`
- Git hash: `1efdaa835178b0ce5034904e29f89f8fc7e0a358`
  - Commit date: `Thu May 22 12:09:45 2025 -0400`

## WER
|Snt|Wrd|Corr|Sub|Del|Ins|Err|
|---|---|---|---|---|---|---|
|588|4963|2479|2394|90|4667|1.441|

## CER
|Snt|Wrd|Corr|Sub|Del|Ins|Err|
|---|---|---|---|---|---|---|
|588|25680|18734|5451|1495|11733|0.727|

## pesq
pesq: 1.101

## stoi
stoi: 0.207

## mcd
mcd: 15.995

## f0rmse
f0rmse: 124.229

## f0corr
f0corr: nan

## utmos
utmos: 3.500

## dns_overall
dns_overall: 3.127

## dns_p808
dns_p808: 3.769

## spk_similarity
spk_similarity: 0.505

## espnet_wer 1.4408623816240178
espnet_wer 1.440

## espnet_cer 0.7273753894080996
espnet_cer 0.727

## Pretrained Models

### espnet/egs2/expresso/tts1/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/

- https://huggingface.co/jihoonk/expresso-vits-espnet2
