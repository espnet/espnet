# LIBRITTS RECIPE

This is the recipe of the English multi-speaker TTS model with [LibriTTS](http://www.openslr.org/60) corpus.

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

# THIRD RESULTS

- Use espeak-ng based G2P

## Environments
- date: `Fri Oct  8 15:48:44 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained Models

### libritts_tts_train_xvector_vits_raw_phn_tacotron_espeak_ng_english_us_vits_train.total_count.ave

<details><summary>Command</summary><div>

```sh
# Prep data directory
./run.sh --stage 1 --stop-stage 1

# Since espeak is super slow, dump phonemized text at first
for dset in train-clean-460 dev-clean test-clean; do
    utils/copy_data_dir.sh data/"${dset}"{,_phn}
    ./pyscripts/utils/convert_text_to_phn.py \
        --nj 32 \
        --g2p espeak_ng_english_us_vits \
        --cleaer tacotron \
        data/"${dset}"{,_phn}/text
done

# Run from stage 2
./run.sh \
    --train_set train-clean-460_phn \
    --valid_set dev-clean_phn \
    --test_sets "dev-clean_phn test-clean_phn" \
    --srctexts "data/train-clean-460_phn/text" \
    --g2p none \
    --cleaner none \
    --stage 2 \
    --min_wav_duration 0.38 \
    --use_xvector true \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_xvector_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05 kHz / 1M iters / Use x-vector / Average the last 10 epochs
- https://zenodo.org/record/5560155


# SECOND RESULTS

- Initial VITS model

## Environments

- date: `Wed Sep 22 22:46:46 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

### libritts_tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --min_wav_duration 0.38 \
    --use_xvector true \
    --ngpu 4 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_xvector_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 22.05 kHz / 1M iters / Use x-vector / Average the last 10 epochs
- https://zenodo.org/record/5521416


# FIRST RESULTS

## Environments
- date: `Sat Jan  2 21:07:44 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.6`
- pytorch version: `pytorch 1.5.1`
- Git hash: `8614318658a497b85dfe5e4aa7e5f6bb06f50a9e`
  - Commit date: `Sat Jan 2 13:23:16 2021 +0900`

## Pretrained Models

### libritts_tts_train_xvector_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4409704

### libritts_tts_train_gst+xvector_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4409702

### libritts_tts_train_xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss
- https://zenodo.org/record/4418754

### libritts_tts_train_gst+xvector_conformer_fastspeech2_transformer_teacher_raw_phn_tacotron_g2p_en_no_space_train.loss
- https://zenodo.org/record/4418774
