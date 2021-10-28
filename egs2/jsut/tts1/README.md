# JSUT RECIPE

This is the recipe of Japanese single female speaker TTS model with [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


# SIXTH RESULTS

- New g2p (`pyopenjtalk_prosody`)

## Environments
- date: `Fri Sep 10 13:04:49 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained models

### jsut_tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_prosody_train.loss.ave
- https://zenodo.org/record/5499026

### jsut_tts_train_transformer_raw_phn_jaconv_pyopenjtalk_prosody_train.loss.ave
- https://zenodo.org/record/5499040

### jsut_tts_train_conformer_fastspeech2_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_prosody_train.loss.ave
- https://zenodo.org/record/5499050

### jsut_tts_train_conformer_fastspeech2_transformer_teacher_raw_phn_jaconv_pyopenjtalk_prosody_train.loss.ave
- https://zenodo.org/record/5499066

### jsut_tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --g2p pyopenjtalk_prosody \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5521354


### jsut_tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_prosody_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --dumpdir dump/44k \
    --expdir exp/44k
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_vits.yaml \
    --g2p pyopenjtalk_prosody \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 44.1khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5521340


# FIFTH RESULTS

- Initial VITS models

## Environments
- date: `Fri Sep  3 21:09:25 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a1`
- pytorch version: `pytorch 1.7.1`
- Git hash: `dee654041cddf80281048b3e7525c1cdafc377ff`
  - Commit date: `Thu Sep 2 14:45:48 2021 +0900`

## Pretrained Models

### jsut_tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --g2p pyopenjtalk_accent_with_pause \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5414980

### jsut_tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --dumpdir dump/44.1k \
    --expdir exp/44.1k
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_vits.yaml \
    --g2p pyopenjtalk_accent_with_pause \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 44.1khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5521360


# FORTH RESULTS

- Use phoneme + accent + pause as the inputs

## Environments
- date: `Wed Jan 13 22:49:20 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.5.1`
- Git hash: `3437dbbceacf5e83c25fe8b426c5d3cbe33333dc`
  - Commit date: `Tue Jan 12 17:28:12 2021 -0500`

## Pretrained Models

### jsut_tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4433194

### jsut_tts_train_transformer_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4433196

### jsut_tts_train_fastspeech_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4436450

### jsut_tts_train_conformer_fastspeech_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4436448

### jsut_tts_train_fastspeech_transformer_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4433200

### jsut_tts_train_conformer_fastspeech_transformer_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause_train.loss.ave
- https://zenodo.org/record/4433198


# THIRD RESULTS

- Use phoneme + accent as the inputs

## Environments
- date: `Mon Dec 21 11:44:18 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.5.1`

## Pretrained Models

### jsut_tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4381098

### jsut_tts_train_transformer_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4381096

### jsut_tts_train_fastspeech_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4381100

### jsut_tts_train_conformer_fastspeech_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4381102

### jsut_tts_train_fastspeech_transformer_teacher_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4391405

### jsut_tts_train_conformer_fastspeech_transformer_teacher_raw_phn_jaconv_pyopenjtalk_accent_train.loss.ave
- https://zenodo.org/record/4391409


# SECOND RESULTS

## Environments
- date: `Thu Oct  1 14:25:19 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.5.1`
- Git hash: `51352aee9ae318640e128a645e722d1f7524edb1`
  - Commit date: `Sat Sep 26 10:35:32 2020 +0900`

## Pretrained Models

### jsut_tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4060508

### jsut_tts_train_fastspeech_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4060510

### jsut_tts_train_conformer_fastspeech_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4060513


# INITIAL RESULTS

## Environments

- date: `Sun Sep 20 19:04:37 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.6.0`
- Git hash: `08b981aa61e6d4fc951af851f0fa4ebb14f00d4c`
  - Commit date: `Sun Sep 20 02:21:47 2020 +0000`

## Pretrained Models

### jsut_tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_train.loss.best
- https://zenodo.org/record/3963886

### jsut_tts_train_fastspeech_raw_phn_jaconv_pyopenjtalk_train.loss.best
- https://zenodo.org/record/3986225

### jsut_tts_train_transformer_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4034121

### jsut_tts_train_fastspeech2_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4032224

### jsut_tts_train_conformer_fastspeech2_raw_phn_jaconv_pyopenjtalk_train.loss.ave
- https://zenodo.org/record/4032246
