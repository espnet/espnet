# RUSLAN RECIPE

This is the recipe of Russian male single speaker TTS model with [RUSLAN Corpus](https://ruslan-corpus.github.io/).

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

## Environments

- date: `Sat Jul 31 10:24:45 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.5.1`
- Git hash: `98691b62c37d04fa9f1f38d76ec13c0591d94832`
  - Commit date: `Fri Jul 30 21:55:52 2021 +0900`

## Pretrained Models

### ruslan_tts_train_transformer_raw_phn_espeak_ng_russian_train.loss.ave
- https://zenodo.org/record/5149485

### ruslan_tts_train_tacotron2_raw_phn_espeak_ng_russian_train.loss.ave
- https://zenodo.org/record/5149493

### ruslan_tts_train_conformer_fastspeech2_raw_phn_espeak_ng_russian_teacher_transformer_train.loss.ave
- https://zenodo.org/record/5150961
