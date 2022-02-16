# CSMSC RECIPE

This is the recipe of Mandarin single female speaker TTS model with [CSMSC](https://www.data-baker.com/#/data/index/source) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


# THIRD RESULTS

- Initial VITS models

## Environments
- date: `Sat Sep  4 19:38:35 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a1`
- pytorch version: `pytorch 1.7.1`
- Git hash: `dee654041cddf80281048b3e7525c1cdafc377ff`
  - Commit date: `Thu Sep 2 14:45:48 2021 +0900`

## Pretrained Models

### csmsc_tts_train_vits_raw_phn_pypinyin_g2p_phone_train.total_count.ave

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
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5499120

### csmsc_tts_train_full_band_vits_raw_phn_pypinyin_g2p_phone_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --win_length null \
    --dumpdir dump/44k \
    --expdir exp/44k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 44.1khz / 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5521404


# SECOND RESULTS

## Environments
- date: `Thu Oct  1 14:25:19 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.5.1`
- Git hash: `51352aee9ae318640e128a645e722d1f7524edb1`
  - Commit date: `Sat Sep 26 10:35:32 2020 +0900`

## Pretrained Models

### csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4060517

### csmsc_tts_train_fastspeech_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4060522

### csmsc_tts_train_conformer_fastspeech_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4060520


# INITIAL RESULTS

## Environments

- date: `Sun Sep 20 19:04:37 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.6.0`
- Git hash: `08b981aa61e6d4fc951af851f0fa4ebb14f00d4c`
  - Commit date: `Sun Sep 20 02:21:47 2020 +0000`

## Pretrained Models

### csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best
- https://zenodo.org/record/3969118

### csmsc_tts_train_fastspeech_raw_phn_pypinyin_g2p_phone_train.loss.best
- https://zenodo.org/record/3986227

### csmsc_tts_train_transformer_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4034125

### csmsc_tts_train_fastspeech2_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4031953

### csmsc_tts_train_conformer_fastspeech2_raw_phn_pypinyin_g2p_phone_train.loss.ave
- https://zenodo.org/record/4031955
