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


# SIXTH RESULTS

- Initial JETS model

## Environments
- date: `Sat May 28 23:00:22 KST 2022`
- python version: `3.7.11 (default, Jul 27 2021, 14:32:16) [GCC 7.5.0]`
- espnet version: `espnet 202204`
- pytorch version: `pytorch 1.10.1`
- Git hash: `047d0c474c18a87c205e566948410be16787e477`
  - Commit date: `Thu May 19 09:50:02 2022 -0400`

## Pretrained Models

### ljspeech_tts_train_jets_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave
- https://huggingface.co/imdanboy/ljspeech_tts_train_jets_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave


# FIFTH RESULTS

- Use espeak-ng based G2P

## Environments
- date: `Fri Oct  8 15:48:44 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.7.1`
- Git hash: `628b46282537ce532d613d6bafb75e826e8455de`
  - Commit date: `Wed Sep 8 13:30:50 2021 +0900`

## Pretrained Models

### ljspeech_tts_train_vits_raw_phn_tacotron_espeak_ng_english_us_vits_train.total_count.ave

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
    --stage 2 \
    --ngpu 4 \
    --g2p none \
    --cleaner none \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

- 1M iters / Average the last 10 epoch models
- https://zenodo.org/record/5555690

### ljspeech_tts_train_tacotron2_raw_phn_tacotron_espeak_ng_english_us_vits_train.loss.ave

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
    --stage 2 \
    --g2p none \
    --cleaner none \
    --train_config ./conf/tuning/train_tacotron2.yaml
```

</div></details>

- Average the best 5 train loss models
- https://zenodo.org/record/5560125

### ljspeech_tts_train_conformer_fastspeech2_raw_phn_tacotron_espeak_ng_english_us_vits_train.loss.ave

<details><summary>Command</summary><div>

```sh
# Use the above tacotron2 model as the teacher
./run.sh \
    --ngpu 1 \
    --stage 7 \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "tr_no_dev_phn dev_phn eval1_phn" \
    --cleaner none \
    --g2p none \
    --train_config ./conf/tuning/train_tacotron2.yaml \
    --tts_exp exp/tts_train_tacotron2_raw_phn_none \
    --inference_args "--use_teacher_forcing true"

# Run fastspeech2 training
./run.sh \
    --train_set tr_no_dev_phn \
    --valid_set dev_phn \
    --test_sets "dev_phn eval1_phn" \
    --stage 5 \
    --g2p none \
    --cleaner none \
    --train_config ./conf/tuning/train_conformer_fastspeech2.yaml \
    --teacher_dumpdir exp/tts_train_tacotron2_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_train_tacotron2_raw_phn_none/decode_use_teacher_forcingtrue_train.loss.ave/stats
```

</div></details>

- Average the best 5 train loss models
- https://zenodo.org/record/5560127


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

<details><summary>Command</summary><div>

```sh
./run.sh \
    --stage 1 \
    --ngpu 4 \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --inference_model train.total_count.ave.pth
```

</div></details>

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
