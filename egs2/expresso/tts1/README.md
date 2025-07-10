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
- date: `Wed Jul 9 21:57:07 UTC 2025`
- python version: `3.10.18 | packaged by conda-forge | (main, Jun 4 2025, 14:45:51)  [GCC 13.3.0]`
- espnet version: `espnet 202503`
- pytorch version: `pytorch 2.4.1+cu124`
- Git hash: `5f146d803dbd998af1f830017b6cf558f0e5ccb2`
  - Commit date: `Mon Jun 9 13:27:27 2025 -0400`

## pesq
pesq: 1.092

## stoi
stoi: 0.205

## mcd
mcd: 8.935

## f0rmse
f0rmse: 86.014

## f0corr
f0corr: 0.166

## utmos
utmos: 3.525

## dns_overall
dns_overall: 3.055

## dns_p808
dns_p808: 3.780

## spk_similarity
spk_similarity: 0.525

## whisper_wer 
whipser_wer 0.149

## whisper_cer 0.7273753894080996
whisper_cer 0.090

## Pretrained Models

### espnet/egs2/expresso/tts1/exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space/

- https://huggingface.co/jihoonk/expresso-vits-espnet2
