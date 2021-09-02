# HUI AUDIO CORPUS GERMAN RECIPE

This is the recipe of German single speaker TTS model with [HUI-audio-corpus-german](https://opendata.iisys.de/datasets.html#hui-audio-corpus-german).

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

- Single female speaker (Hokuspokus)

## Environments

- date: `Sun Aug  1 10:40:12 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.5.1`
- Git hash: `9e6803d2f6c68e951268d0a4d23460737d9fdd39`
  - Commit date: `Sun Aug 1 10:14:48 2021 +0900`

## Pretrained Models

### hui_acg_tts_train_transformer_raw_hokuspokus_phn_espeak_ng_german_train.loss.ave
- https://zenodo.org/record/5150954

### hui_acg_tts_train_tacotron2_raw_hokuspokus_phn_espeak_ng_german_train.loss.ave
- https://zenodo.org/record/5150957

### hui_acg_tts_train_conformer_fastspeech2_raw_hokuspokus_phn_espeak_ng_german_teacher_transformer_train.loss.ave
- https://zenodo.org/record/5150959
