# ESPnet2 TTS Recipe TEMPLATE

This is a template of TTS recipe for ESPnet2.

## Table of Contents

* [ESPnet2 TTS Recipe TEMPLATE](#espnet2-tts-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Wav dump or Feature extraction](#2-wav-dump-or-feature-extraction)
    * [3\. Removal of long / short data](#3-removal-of-long--short-data)
    * [4\. Token list generation](#4-token-list-generation)
    * [5\. TTS statistics collection](#5-tts-statistics-collection)
    * [6\. TTS training](#6-tts-training)
    * [7\. TTS decoding](#7-tts-decoding)
    * [8\-9\. (Optional) Pack results for upload](#8-9-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [FastSpeech training](#fastspeech-training)
    * [FastSpeech2 training](#fastspeech2-training)
  * [Supported text frontend](#supported-text-frontend)
  * [Supported text cleaner](#supported-text-cleaner)
  * [Supported Models](#supported-models)
    * [Single speaker model](#single-speaker-model)
    * [Multi speaker model](#multi-speaker-model)
  * [FAQ](#faq)
    * [How to change minibatch size in training?](#how-to-change-minibatch-size-in-training)
    * [How to make a new recipe for my own dataset?](#how-to-make-a-new-recipe-for-my-own-dataset)
    * [How to add a new g2p module?](#how-to-add-a-new-g2p-module)
    * [How to add a new cleaner module?](#how-to-add-a-new-cleaner-module)
    * [How to use trained model in python?](#how-to-use-trained-model-in-python)
    * [How to finetune the pretrained model?](#how-to-finetune-the-pretrained-model)
    * [How to add a new model?](#how-to-add-a-new-model)
    * [How to test my model with an arbitrary given text?](#how-to-test-my-model-with-an-arbitrary-given-text)

## Recipe flow

TTS recipe consists of 9 stages.

### 1. Data preparation

Data preparation stage.
It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Wav dump or Feature extraction

Feature extraction stage.
The processing in this stage is changed according to `--feats_type` option (Default: `feats_type=raw`).
In the case of `feats_type=raw`, reformat `wav.scp` in date directories.
In the other cases (`feats_type=fbank` and `feats_type=stft`), feature extraction with Librosa will be performed.
Since the performance is almost the same, we recommend using `feats_type=raw`.

### 3. Removal of long / short data

Processing stage to remove long and short utterances from the training and validation data.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

### 4. Token list generation

Token list generation stage.
It generates token list (dictionary) from `srctexts`.
You can change the tokenization type via `--token_type` option.
`token_type=char` and `token_type=phn` are supported.
If `--cleaner` option is specified, the input text will be cleaned with the specified cleaner.
If `token_type=phn`, the input text will be converted with G2P module specified by `--g2p` option.

See also:
- [Supported text cleaner](#supported-text-cleaner).
- [Supported text frontend](#supported-text-frontend).

### 5. TTS statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training data).

### 6. TTS training

TTS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 7. TTS decoding

TTS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 8-9. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

## How to run

Here, we show the procedure to run the recipe using `egs2/ljspeech/tts1`.

Move on the recipe directory.
```sh
$ cd egs2/ljspeech/tts1
```

Modify `LJSPEECH` variable in `db.sh` if you want to change the download directory.
```sh
$ vim db.sh
```

Modify `cmd.sh` and `conf/*.conf` if you want to use job scheduler.
See the detail in [using job scheduling system](https://espnet.github.io/espnet/parallelization.html).
```sh
$ vim cmd.sh
```

Run `run.sh`, which conducts all of the stages explained above.
```sh
$ ./run.sh
```
As a default, we train Tacotron2 (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/        # validation set
│   ├── eval1/      # evaluation set
│   ├── token_list/ # token list (dictionary)
│   └── tr_no_dev/  # training set
├── dump/ # feature dump directory
│   └── raw/
│       ├── dev/       # validation set
│       ├── eval1/     # evaluation set
│       ├── srctexts   # text to create token list
│       └── tr_no_dev/ # training set
└── exp/ # experiment directory
    ├── tts_stats_raw_phn_tacotron_g2p_en_no_space # statistics
    └── tts_train_raw_phn_tacotron_g2p_en_no_space # model
        ├── att_ws/                 # attention plot during training
        ├── tensorboard/            # tensorboard log
        ├── images/                 # plot of training curves
        ├── decode_train.loss.best/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval1/ # evaluation set
        │        ├── att_ws/      # attention plot in decoding
        │        ├── probs/       # stop probability plot in decoding
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── durations    # duration of each input tokens
        │        ├── feats_type   # feature type
        │        ├── focus_rates  # focus rate
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_5best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```
In decoding, we use Griffin-Lim for waveform generation.
If you want to combine with neural vocoder, please use [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).
See the detail in [decoding with ESPnet-TTS model's feature](https://github.com/kan-bayashi/ParallelWaveGAN#decoding-with-espnet-tts-models-features).

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
$ ./run.sh --stage 3 --stop-stage 3
```
This might helps you to understand each stage's processing and directory structure.

### FastSpeech training

If you want to train FastSpeech, additional steps with the teacher model are needed.
Please make sure you already finished the training of the teacher model (Tacotron2 or Transformer-TTS).

First, decode all of data including training, validation, and evaluation set.
```sh
# specify teacher model directory via --tts_exp option
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --test_sets "tr_no_dev dev eval1"
```
This will generate `durations` for training, validation, and evaluation sets in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.best`.

Then, you can train FastSpeech by specifying the directory including `durations` via `--teacher_dumpdir` option.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_train.loss.best
```

In the above example, we use generated mel-spectrogram as the target, which is known as knowledge distillation training.
If you want to use groundtruth mel-spectrogram as the target, we need to use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```
You can get the groundtruth aligned durations in `exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.best`.

Then, you can train FastSpeech without knowledge distillation.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.best
```

### FastSpeech2 training

The procedure is almost the same as FastSpeech but we **MUST** use teacher forcing in decoding.
```sh
$ ./run.sh --stage 7 \
    --tts_exp exp/tts_train_raw_phn_tacotron_g2p_en_no_space \
    --inference_args "--use_teacher_forcing true" \
    --test_sets "tr_no_dev dev eval1"
```

To train FastSpeech2, we use additional feature (F0 and energy).
Therefore, we need to start from `stage 5` to calculate additional statistics.
```sh
$ ./run.sh --stage 5 \
    --train_config conf/tuning/train_fastspeech.yaml \
    --teacher_dumpdir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.best \
    --tts_stats_dir exp/tts_train_raw_phn_tacotron_g2p_en_no_space/decode_use_teacher_forcingtrue_train.loss.best/stats \
    --write_collected_feats true
```
where `--tts_stats_dir` is the option to specify the directory to dump Statistics, and `--write_collected_feats` is the option to dump features in statistics calculation.
The use of `--write_collected_feats` is optional but it helps to accelerate the training.

## Supported text frontend

You can change via `--g2p` option in `tts.sh`.

- `g2p_en`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - e.g. `Hello World` -> `HH AH0 L OW1 <space> W ER1 L D`
- `g2p_en_no_space`: [Kyubyong/g2p](https://github.com/Kyubyong/g2p)
    - Same G2P but do not use word separator
    - e.g. `Hello World` -> `HH AH0 L OW1 W ER1 L D`
- `pyopenjtalk`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - e.g. `こんにちは` -> `k o N n i ch i w a`
- `pyopenjtalk_kana`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - Use kana instead of phoneme
    - e.g. `こんにちは` -> `コンニチワ`
- `pypinyin`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - e.g. `卡尔普陪外孙玩滑梯。` -> `ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 。`
- `pypinyin_phone`: [mozillanzg/python-pinyin](https://github.com/mozillazg/python-pinyin)
    - Separate into first and last parts
    - e.g. `卡尔普陪外孙玩滑梯。` -> `k a3 er3 p u3 p ei2 wai4 s un1 uan2 h ua2 t i1 。`

## Supported text cleaner

You can change via `--cleaner` option in `tts.sh`.

- `none`: No text cleaner.
- `tacotron`: [keithito/tacotron](https://github.com/keithito/tacotron)
    - e.g.`"(Hello-World);  & jr. & dr."` ->`HELLO WORLD, AND JUNIOR AND DOCTOR`
- `jaconv`: [kazuhikoarase/jaconv](https://github.com/kazuhikoarase/jaconv)
    - e.g. `”あらゆる”　現実を　〜　’すべて’ 自分の　ほうへ　ねじ曲げたのだ。"` -> `"あらゆる" 現実を ー \'すべて\' 自分の ほうへ ねじ曲げたのだ。`

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `tts.sh`.

### Single speaker model

- [Tacotron 2](https://arxiv.org/abs/1712.05884)
- [Transformer-TTS](https://arxiv.org/abs/1809.08895)
- [FastSpeech](https://arxiv.org/abs/1905.09263)
- [FastSpeech2](https://arxiv.org/abs/2006.04558) ([FastPitch](https://arxiv.org/abs/2006.06873))
- [Conformer](https://arxiv.org/abs/2005.08100)-based FastSpeech / FastSpeech2

You can find example configs of the above models in [`egs2/ljspeech/tts1/conf/tuning`](../../ljspeech/tts1/conf/tuning).

### Multi speaker model

- [GST + Tacotron2](https://arxiv.org/abs/1803.09017)
- GST + Transformer-TTS
- GST + FastSpeech
- GST + FastSpeech2
- GST + Conformer-based FastSpeech / FastSpeech2

You can find example configs of the above models in [`egs2/vctk/tts1/conf/tuning`](../../vctk/tts1/conf/tuning).

## FAQ

### How to change minibatch size in training?

See [change mini-batch type](https://espnet.github.io/espnet/espnet2_training_option.html#change-mini-batch-type).
As a default, we use `batch_type=numel` and `batch_bins` instead of `batch_size` to enable us to use dynamic batch size.
See the following config as an example.
https://github.com/espnet/espnet/blob/96b2fd08d4fd9276aabd7ad41ec5e02a88b30958/egs2/ljspeech/tts1/conf/tuning/train_tacotron2.yaml#L61-L62

### How to make a new recipe for my own dataset?

See [how to make/port new recipe](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#how-to-makeport-new-recipe).

### How to add a new `g2p` module?

Update `espnet2/text/phoneme_tokenizer.py` to add new module.
Then, add new choice in the argument parser of `espnet2/bin/tokenize_text.py` and `espnet2/tasks/tts.py`.

### How to add a new `cleaner` module?

Update `espnet2/text/cleaner.py` to add new module.
Then, add new choice in the argument parser of `espnet2/bin/tokenize_text.py` and `espnet2/tasks/tts.py`.

### How to use trained model in python?

See [use a pretrained model for inference](https://github.com/espnet/espnet_model_zoo#use-a-pretrained-model-for-inference).

### How to finetune the pretrained model?

Please use `--pretrain_path` and `--pretrain_key` options in training config (`*.yaml`).
See the usage in [abs_task.py](https://github.com/espnet/espnet/blob/3cc59a16c3655f3b39dc2ae19ffafa7bfac879bf/espnet2/tasks/abs_task.py#L1040-L1054).

### How to add a new model?

Under construction.

### How to test my model with an arbitrary given text?

See Google Colab demo notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espnet/notebook/blob/master/espnet2_tts_realtime_demo.ipynb)
