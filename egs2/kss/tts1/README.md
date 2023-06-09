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
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


# FOURTH RESULTS

- Initial JETS model

```sh
# Run with the following command for jets:
./run.sh \
    --tts_task gan_tts \
    --fs 24000 \
    --fmin 0 \
    --fmax null \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --train_config conf/tuning/train_jets.yaml \
    --token_type phn \
    --g2p g2pk \
    --cleaner null
```

## Environments
- date: `Mon May 30 00:51:37 KST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16) [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.10.1`
- Git hash: `047d0c474c18a87c205e566948410be16787e477`
  - Commit date: `Thu May 19 09:50:02 2022 -0400`

## Pretrained models

### kss_tts_train_jets_raw_phn_null_g2pk_train.total_count.ave
- https://huggingface.co/imdanboy/kss_tts_train_jets_raw_phn_null_g2pk_train.total_count.ave


# THIRD RESULTS
- Applied with `korean_jaso` and `korean_cleaner`
- Sampling frequency of 44,100 Hz
- VITS configuration applied.

```sh
# Run with the following command for vits:
./run.sh \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --fs 44100 \
    --fmin 80 \
    --fmax 22050 \
    --n_mels 120 \
    --n_fft 2048 \
    --n_shift 512 \
    --win_length 2048 \
    --train_config conf/tuning/train_full_band_vits.yaml \
    --inference_config conf/tuning/decode_vits.yaml \
    --token_type phn \
    --g2p korean_jaso \
    --cleaner korean_cleaner
```

## Environments
- date: `Wed Oct 13 16:56:45 KST 2021`
- python version: `3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.4a1`
- chainer version: `chainer 7.8.0`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `e2c8c30580caf010a957b278df6083bcab14117e`
  - Commit date: `Tue Oct 12 15:25:39 2021 +0900`

## Pretrained models

### kss_vits_vits_44100_train.total_count.best, fs=44100, lang=ko
- https://zenodo.org/record/5563406


# SECOND RESULTS
- New G2P of `korean_jaso` (korean grapheme-based tokenizer)
- New cleaner of `korean_cleaner` (basic one, not sophisticated)
- Sampling frequency of 44,100 Hz

```sh
# Run with the following command for tacotron2:
./run.sh \
    --fs 44100 \
    --fmin 80 \
    --fmax 22050 \
    --n_mels 120 \
    --n_fft 2048 \
    --n_shift 512 \
    --win_length 2048 \
    --train_config conf/tuning/train_tacotron2.yaml \
    --inference_config conf/tuning/decode_tacotron2.yaml \
    --token_type phn \
    --g2p korean_jaso \
    --cleaner korean_cleaner
```

## Environments
- date: `Tue Sep 14 13:39:16 KST 2021`
- python version: `3.9.5 (default, Jun  4 2021, 12:28:51)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a2`
- chainer version: `chainer 7.8.0`
- pytorch version: `pytorch 1.9.0+cu102`
- Git hash: `97b9dad4dbca71702cb7928a126ec45d96414a3f`
  - Commit date: `Mon Sep 13 22:55:04 2021 +0900`

## Pretrained models

### kss_tts_train_tacotron2_raw_phn_korean_cleaner_korean_jaso_train.loss.ave, fs=44100
- https://zenodo.org/record/5508413


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
