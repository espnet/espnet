# KSS RECIPE

This is the recipe of Korean female single speaker TTS model with [KSS dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset).

Before running the recipe, please download from https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset.  
Then, edit 'KSS' in `db.sh` and locate unzipped dataset as follows:

```bash
$ vim db.sh
KSS=/path/to/kss

$ tree -L 1 /path/to/kss
/path/to/kss
├── 1
├── 2
├── 3
├── 4
└── transcript.v.1.4.txt
```

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# INITIAL RESULTS

## Environments
- date: `Tue Aug  3 15:23:52 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.0`
- pytorch version: `pytorch 1.5.1`
- Git hash: `0b17d7081defbe2d3b840fdbf488007860b3a6c3`
  - Commit date: `Mon Aug 2 21:35:48 2021 -0400`

## Pretrained models

### kss_tts_train_transformer_raw_phn_g2pk_no_space_train.loss.ave
- https://zenodo.org/record/5154791

### kss_tts_train_tacotron2_raw_phn_g2pk_no_space_train.loss.ave
- https://zenodo.org/record/5154795

### kss_tts_train_conformer_fastspeech2_raw_phn_g2pk_no_space_teacher_transformer_train.loss.ave
- https://zenodo.org/record/5154835
