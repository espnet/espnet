# LIBRITTS RECIPE

This is the recipe of the English multi-speaker TTS model with [LibriTTS](http://www.openslr.org/60) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train with X-vector](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-x-vector-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# FIRST RESULTS

## Environments
- date: `Sat Jan  2 21:07:44 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.5.1`
- Git hash: `8614318658a497b85dfe5e4aa7e5f6bb06f50a9e`
  - Commit date: `Sat Jan 2 13:23:16 2021 +0900`

## Pretrained Models

### libritts_tts_train_xvector_trasnformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4409704

### libritts_tts_train_gst+xvector_trasnformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4409702

### libritts_tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss
- https://zenodo.org/record/4418754

### libritts_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss
- https://zenodo.org/record/4418774
