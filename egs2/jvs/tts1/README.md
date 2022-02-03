# JVS RECIPE

This is the recipe of the adaptation with Japanese single speaker in [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) corpus.

This recipe assumes the use of pretrained model.
Please follow the usage to perform fine-tuning with pretrained model.
See the following pages before asking the question:
- [ESPnet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [ESPnet2 TTS FAQ](../../TEMPLATE/tts1/README.md#faq)

# HOW TO RUN

- [AR model case (Tacotron2 / Transformer)](#ar-model-case-tacotron2--transformer)
- [Non-AR model case (FastSpeech / FastSpeech2)](#non-ar-model-case-fastspeech--fastspeech2)
- [VITS case](#vits-case)

## AR model case (Tacotron2 / Transformer)

Here, we show the procedure of the fine-tuning using Tacotron2, which was pretrained on [JSUT](../../jsut/tts1) corpus using `pyopenjtalk_accent_with_pause` G2P.

### 1. Run the recipe until stage 5

```sh
# From data preparation to statistics calculation
$ ./run.sh --stop-stage 5 --g2p pyopenjtalk_accent_with_pause
```

The detail of stage 1-5 can be found in [`Recipe flow`](../../TEMPLATE/tts1/README.md#recipe-flow).

### 2. Download pretrained model

Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/jsut_tacotron2_accent_with_pause
```

You can find the other pretrained models in [ESPnet model zoo](https://github.com/espnet/espnet_model_zoo/blob/master/espnet_model_zoo/table.csv).

### 3. Replace token list with pretrained model's one

Since we use the same language data for fine-tuning, we need to use the token list of the pretrained model instead of that of data for fine-tuning.
The downloaded pretrained model has `tokens_list` in the config, so first we create `tokens.txt` (`token_list`) from the config.

```sh
$ pyscripts/utils/make_token_list_from_config.py downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/config.yaml

# tokens.txt is created in model directory
$ ls downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause
config.yaml  images  tokens.txt  train.loss.ave_5best.pth
```

Let us replace the `tokens.txt` with pretrained model's one.
```sh
# Make backup (Rename -> *.bak)
$ mv dump/token_list/phn_jaconv_pyopenjtalk_accent_with_pause/tokens.{txt,txt.bak}
# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -s $(pwd)/downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/tokens.txt dump/token_list/phn_jaconv_pyopenjtalk_accent_with_pause
```

### 4 (Optional). Replace statistics with pretrained model's one

Sometimes, using the feature statistics of the pretrained models is better than using that of adaptation data.
This is an optional step, so you can skip if you use the original statistics.

```sh
# Make backup (Rename -> *.bak)
$ mv exp/tts_stats_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train/feats_stats.{npz,npz.bak}
# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -s $(pwd)/downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_stats_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train/feats_stats.npz exp/tts_stats_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train
```

### 5. Run fine-tuning

Run the recipe from stage 6.

You need to specify `--init_param` for `--train_args` to load pretrained parameters (Or you can write them in `*.yaml` config).
Here `--init_param /path/to/model.pth:a:b` represents loading "a" parameters in model.pth into "b", and `:tts:tts` means load parameters except for the feature normalizer.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 6 \
    --g2p pyopenjtalk_accent_with_pause \
    --train_config conf/tuning/finetune_tacotron2.yaml \
    --train_args "--init_param downloads/0afe7c220cac7d9893eea4ff1e4ca64e/exp/tts_train_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause
```

For more complex loading of pretrained parameters, please check [`How to load pretrained model?`](../../TEMPLATE/tts1/README.md#how-to-load-the-pretrained-model) For example, if you want to perform fine-tuning of English model with Japanese data, you may want to load the network except for the token embedding layer.

## Non-AR model case (FastSpeech / FastSpeech2)

To finetune non-AR models, we need to preapre `durations` file.
Therefore, at first, please finish the finetuning of AR models by the above steps.

Here, we show the procedure of FastSpeech2 fine-tuning with the above fine-tuened tacotron2 as the teacher.

### 1. Prepare durations file using the adapted AR model

First, prepare the `durations` for all sets by running AR model inference with teacher forcing.

```sh
$ ./run.sh \
    --stage 7 \
    --g2p pyopenjtalk_accent_with_pause \
    --tts_exp exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "jvs010_tr_no_dev jvs010_dev jvs010_eval1"
```

You can find `durations` files in `exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave/*`.

### 2. Download pretrained model

Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/jsut_fastspeech2_accent_with_pause
```

Please make sure this model used the same `token_list` as the teacher AR model.

### 3. Run fine-tuning

Here we skip the replacement of the statistics (Of course you can do it).
And we assume that `tokens.txt` is already replaced in AR model fine-tuning.

Since fastspeech2 requires extra feature calculation, run from stage 5.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 5 \
    --g2p pyopenjtalk_accent_with_pause \
    --write_collected_feats true \
    --teacher_dumpdir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave \
    --tts_stats_dir exp/tts_finetune_tacotron2_raw_phn_jaconv_pyopenjtalk_accent_with_pause/decode_use_teacher_forcingtrue_train.loss.ave/stats \
    --train_config conf/tuning/finetune_fastspeech2.yaml \
    --train_args "--init_param downloads/0293a01e429a84a604304bf06f2cc0b0/exp/tts_train_fastspeech2_tacotron2_teacher_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.loss.ave_5best.pth:tts:tts" \
    --tag finetune_fastspeech2_raw_phn_jaconv_pyopenjtalk_accent_with_pause
```

## VITS case

In the case of VITS, please be careful about the sampling rate.
As a default, vits used 22.05 khz (but this recipe default is 24khz).

### 1. Run the recipe until stage 5 with 22.05khz setup

```sh
# Here we changed root dumpdir from dump -> dump/22k and
# different g2p to match with the pretrained model.
# `min_wav_duration` is need to filter out less than 0.38 sec (~=8,192 / 22,050).
$ ./run.sh \
    --stage 1 \
    --stop-stage 5 \
    --g2p pyopenjtalk_accent_with_pause \
    --min_wav_duration 0.38 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/finetune_vits.yaml
```

### 2. Download pretrained model

Download pretrained model from ESPnet model zoo here.
If you have your own pretrained model, you can skip this step.

```sh
$ . ./path.sh
$ espnet_model_zoo_download --unpack true --cachedir downloads kan-bayashi/jsut_vits_accent_with_pause
```

### 3. Replace token list with pretrained model's one

Since we use the same language data for fine-tuning, we need to use the token list of the pretrained model instead of that of data for fine-tuning.
The downloaded pretrained model has `tokens_list` in the config, so first we create `tokens.txt` (`token_list`) from the config.

```sh
$ pyscripts/utils/make_token_list_from_config.py downloads/f3698edf589206588f58f5ec837fa516/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_with_accent/config.yaml

# tokens.txt is created in model directory
$ ls downloads/f3698edf589206588f58f5ec837fa516/exp/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause
config.yaml  images  train.total_count.ave_10best.pth
```

Let us replace the `tokens.txt` with pretrained model's one.
```sh
# Make backup (Rename -> *.bak)
$ mv dump/22k/token_list/phn_jaconv_pyopenjtalk_accent_with_pause/tokens.{txt,txt.bak}
# Make symlink to pretrained model's one (Just copy is also OK)
$ ln -s $(pwd)/downloads/f3698edf589206588f58f5ec837fa516/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/tokens.txt dump/22k/token_list/phn_jaconv_pyopenjtalk_accent_with_pause
```

### 4. Run fine-tuning

Run from stage 6.

```sh
# Recommend using --tag to name the experiment directory
$ ./run.sh \
    --stage 6 \
    --g2p pyopenjtalk_accent_with_pause \
    --min_wav_duration 0.38 \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --dumpdir dump/22k \
    --win_length null \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/finetune_vits.yaml \
    --train_args "--init_param downloads/f3698edf589206588f58f5ec837fa516/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.total_count.ave_10best.pth:tts:tts" \
    --tag finetune_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause
```

# VITS EXAMPLE RESULTS

## Environments
- date: `Fri Sep  3 21:09:25 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.10.3a1`
- pytorch version: `pytorch 1.7.1`
- Git hash: `dee654041cddf80281048b3e7525c1cdafc377ff`
  - Commit date: `Thu Sep 2 14:45:48 2021 +0900`

## Pretrained Models

### kan-bayashi/jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause_latest
- 22.05 khz / jvs001 (male) / 50k iters
- https://zenodo.org/record/5432540

### kan-bayashi/jvs_tts_finetune_jvs010_jsut_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause_latest
- 22.05 khz / jvs010 (female) / 50k iters
- https://zenodo.org/record/5432566
