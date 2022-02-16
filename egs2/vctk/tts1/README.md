# VCTK RECIPE

This is the recipe of the English multi-speaker TTS model with [VCTK](http://www.udialogue.org/download/cstr-vctk-corpus.html) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train with X-vector](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-x-vector-training)
- [How to train with speaker ID](../../TEMPLATE/tts1/README.md#multi-speaker-model-with-speaker-id-embedding-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


# FORTH RESULTS

- Use espeak-ng based G2P

## Environments
- date: `Fri Oct  8 15:48:44 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained Models

### vctk_tts_train_multi_spk_vits_raw_phn_tacotron_espeak_ng_english_us_vits_train.total_count.ave

<details><summary>Command</summary><div>

```sh
# Prep data directory
./run.sh --stage 1 --stop-stage 1

# Since espeak is super slow, dump phonemized text at first
for dset in tr_no_dev dev eval1; do
    utils/copy_data_dir.sh data/"${dset}"{,_phn}
    ./pyscripts/utils/convert_text_to_phn.py \
        --nj 32 \
        --g2p espeak_ng_english_us_vits \
        --cleaer tacotron \
        data/"${dset}"{,_phn}/text
done

# Run from stage 2
./run.sh \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "dev_phn eval1_phn" \
    --srctexts "data/tr_no_dev_phn/text" \
    --g2p none \
    --cleaner none \
    --stage 2 \
    --use_sid true \
    --min_wav_duration 0.38 \
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
    --train_config ./conf/tuning/train_multi_spk_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05 kHz / 1M iters / Use speaker ID (one-hot) / Averaged the last 5 epochs
- https://zenodo.org/record/5560132

### vctk_tts_train_xvector_vits_raw_phn_tacotron_espeak_ng_english_us_vits_train.total_count.ave

<details><summary>Command</summary><div>

```sh
# Prep data directory
./run.sh --stage 1 --stop-stage 1

# Since espeak is super slow, dump phonemized text at first
for dset in tr_no_dev dev eval1; do
    utils/copy_data_dir.sh data/"${dset}"{,_phn}
    ./pyscripts/utils/convert_text_to_phn.py \
        --nj 32 \
        --g2p espeak_ng_english_us_vits \
        --cleaer tacotron \
        data/"${dset}"{,_phn}/text
done

# Run from stage 2
./run.sh \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "dev_phn eval1_phn" \
    --srctexts "data/tr_no_dev_phn/text" \
    --g2p none \
    --cleaner none \
    --stage 2 \
    --use_xvector true \
    --min_wav_duration 0.38 \
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
    --train_config ./conf/tuning/train_xvector_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05 kHz / 1M iters / Use X-vector / Averaged the last 5 epochs
- https://zenodo.org/record/5560146

### vctk_tts_train_full_band_multi_spk_vits_raw_phn_tacotron_espeak_ng_english_us_vits_train.total_count.ave

<details><summary>Command</summary><div>

```sh
# Prep data directory
./run.sh --stage 1 --stop-stage 1

# Since espeak is super slow, dump phonemized text at first
for dset in tr_no_dev dev eval1; do
    utils/copy_data_dir.sh data/"${dset}"{,_phn}
    ./pyscripts/utils/convert_text_to_phn.py \
        --nj 32 \
        --g2p espeak_ng_english_us_vits \
        --cleaer tacotron \
        data/"${dset}"{,_phn}/text
done

# Run from stage 2
./run.sh \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "dev_phn eval1_phn" \
    --srctexts "data/tr_no_dev_phn/text" \
    --g2p none \
    --cleaner none \
    --stage 2 \
    --use_sid true \
    --min_wav_duration 0.38 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --dumpdir dump/44k \
    --expdir exp/44k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_multi_spk_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 44.1 kHz / 1M iters / Use speaker ID (one-hot) / Averaged the last 5 epochs
- https://zenodo.org/record/5560148


# THIRD RESULTS

- Initial VITS models

## Environments
- date: `Sat Sep 11 09:52:43 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained Models

### vctk_tts_train_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --use_sid true \
    --min_wav_duration 0.38 \
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
    --train_config ./conf/tuning/train_multi_spk_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05 kHz / 1M iters / Use speaker ID (one-hot) / Averaged the last 10 epochs
- https://zenodo.org/record/5500759

### vctk_tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --use_sid true \
    --min_wav_duration 0.38 \
    --ngpu 4 \
    --fs 44100 \
    --n_fft 2048 \
    --n_shift 512 \
    --dumpdir dump/44k \
    --expdir exp/44k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_full_band_multi_spk_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 44.1 kHz / 1M iters / Use speaker ID (one-hot) / Averaged the last 10 epochs
- https://zenodo.org/record/5521431


# SECOND RESULTS

- Use X-vector as the speaker embedding

## Environments

- date: `Fri Dec 25 15:32:06 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.8.0`
- pytorch version: `pytorch 1.5.1`
- Git hash: `c86bb088061641e9fba08310b4e6826e5e819f54`
  - Commit date: `Fri Dec 25 14:45:38 2020 +0900`

## Pretrained Models

### vctk_tts_train_xvector_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4393279

### vctk_tts_train_gst+xvector_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4393277

### vctk_tts_train_xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4394600

### vctk_tts_train_gst+xvector_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4394598

### vctk_tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4394602

### vctk_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4394608


# INITIAL RESULTS

## Environments

- date: `Sun Sep 20 19:04:37 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.6.0`
- Git hash: `08b981aa61e6d4fc951af851f0fa4ebb14f00d4c`
  - Commit date: `Sun Sep 20 02:21:47 2020 +0000`

## Pretrained Models

### vctk_tts_train_gst_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.best
- https://zenodo.org/record/3986237

### vctk_tts_train_gst_fastspeech_raw_phn_tacotron_g2p_en_no_space_train.loss.best
- https://zenodo.org/record/3986241

### vctk_tts_train_gst_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4037456

### vctk_tts_train_gst_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4036266

### vctk_tts_train_gst_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4036264
