# LJSPEECH RECIPE

This is the recipe of English single female speaker TTS model with [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus.

See the following pages for the usage:
- [How to run the recipe](../../TEMPLATE/tts1/README.md#how-to-run)
- [How to train FastSpeech](../../TEMPLATE/tts1/README.md#fastspeech-training)
- [How to train FastSpeech2](../../TEMPLATE/tts1/README.md#fastspeech2-training)
- [How to train VITS](../../TEMPLATE/tts1/README.md#vits-training)
- [How to train joint text2wav](../../TEMPLATE/tts1/README.md#joint-text2wav-training)

See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)


# FORTH RESULTS

- Initial joint training models

## Environments
- date: `Fri Sep 10 13:04:49 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained models

### ljspeech_tts_train_joint_conformer_fastspeech2_hifigan_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave
- Joint training from scrath
- 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5498487

### ljspeech_tts_finetune_joint_conformer_fastspeech2_hifigan_initilized_discriminator_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave
- Joint finetuning with the initialized discriminator
- 0.5M iters / Average the last 5 epoch models
- https://zenodo.org/record/5498497

### ljspeech_tts_finetune_joint_conformer_fastspeech2_hifigan_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave
- Joint finetuning with pretrained text2mel, vocoder and discriminator
- 0.5M iters / Average the last 5 epoch models
- https://zenodo.org/record/5498896


# THIRD RESULTS

- Initial VITS model

## Environments
- date: `Sat Sep  4 19:38:35 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a1`
- pytorch version: `pytorch 1.7.1`
- Git hash: `dee654041cddf80281048b3e7525c1cdafc377ff`
  - Commit date: `Thu Sep 2 14:45:48 2021 +0900`

## Pretrained Models

### ljspeech_tts_train_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave
- 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5443814


# SECOND RESULTS

## Environments
- date: `Thu Oct  1 14:25:19 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.5.1`
- Git hash: `51352aee9ae318640e128a645e722d1f7524edb1`
  - Commit date: `Sat Sep 26 10:35:32 2020 +0900`

## Pretrained Models

### ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4060524

### ljspeech_tts_train_fastspeech_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4060526

### ljspeech_tts_train_conformer_fastspeech_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4060528


# INITIAL RESULTS

## Environments

- date: `Sun Sep 20 19:04:37 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.3`
- pytorch version: `pytorch 1.6.0`
- Git hash: `08b981aa61e6d4fc951af851f0fa4ebb14f00d4c`
  - Commit date: `Sun Sep 20 02:21:47 2020 +0000`

## Pretrained Models

### ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train.loss.best
- https://zenodo.org/record/3989498

### ljspeech_tts_train_fastspeech_raw_phn_tacotron_g2p_en_no_space_train.loss.best
- https://zenodo.org/record/3986231

### ljspeech_tts_train_transformer_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4039194

### ljspeech_tts_train_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4036272

### ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_g2p_en_no_space_train.loss.ave
- https://zenodo.org/record/4036268
