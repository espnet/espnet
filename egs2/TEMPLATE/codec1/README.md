# ESPnet2 Codec Recipe TEMPLATE

This is a template of Codec recipe for ESPnet2.

## Table of Contents

* [ESPnet2 Codec Recipe TEMPLATE](#espnet2-codec-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Wav dump / Embedding preparation](#2-wav-dump--embedding-preparation)
    * [3\. Removal of long / short data](#3-removal-of-long--short-data)
    * [4\. Codec statistics collection](#4-codec-statistics-collection)
    * [5\. Codec training](#5-codec-training)
    * [6\. Codec decoding](#6-codec-decoding)
    * [7\. Codec Scoring](#7-codec-scoring)
    * [8\-9\. (Optional) Pack results for upload](#8-9-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [Basic training](#basic-training)
    * [Scoring](#scoring)
  * [Supported Models](#supported-models)
  * [FAQ](#faq)

## Recipe flow

Codec recipe consists of 9 stages.

### 1. Data preparation

Data preparation stage.
You have two methods to generate the data:

#### ESPnet format:

It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets.

Noted that since we usually just need waveform to train the model, we can just use `wav.scp` to train a model.
However, if you would like to use additional information (transcription, speaker information) for evaluation, we may want to have full-kaldi-style supports.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)


### 2. Wav dump / Embedding preparation

Wav dumping stage.
This stage reformats `wav.scp` in data directories.

If you specify kaldi, then we additionally extract mfcc features and vad decision.

### 3. Removal of long / short data

Processing stage to remove long and short utterances from the training and validation data.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

### 4. Codec statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training data) if needed.

### 5. Codec training

Codec model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 6. Codec decoding

Codec model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 7. Codec Scoring

Codec model scoring stage.
The scoring is supported by [VERSA](https://github.com/shinjiwlab/versa).
You can change the scoring setting via `--scoring_config` and `--scoring_args`.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [VERSA documents](https://github.com/shinjiwlab/versa)

### 8. (Optional) Pack results for upload

Packing stage.
It packs the trained model files as a preparation for uploading to Hugging Face.

### 9. (Optional) Upload model to Hugging Face

Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to run

Here, we show the procedure to run the recipe using `egs2/libritts/codec1`.

Move on the recipe directory.
```sh
$ cd egs2/libritts/codec1
```

Modify `libritts` variable in `db.sh` if you want to change the download directory.
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
│   ├── dev/        # validation set
│   ├── eval1/      # evaluation set
│   └── tr_no_dev/  # training set
├── dump/ # feature dump directory
│   └── raw/
│       ├── org/
│       │    ├── tr_no_dev/ # training set before filtering
│       │    └── dev/       # validation set before filtering
│       ├── eval1/     # evaluation set
│       ├── dev/       # validation set after filtering
│       └── tr_no_dev/ # training set after filtering
└── exp/ # experiment directory
    ├── codec_stats_raw_phn_tacotron_g2p_en_no_space # statistics
    └── codec_train_raw_phn_tacotron_g2p_en_no_space # model
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── decode_train.loss.ave/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval1/ # evaluation set
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── feats_type   # feature type
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_5best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
...
$ ./run.sh --stage 8 --stop-stage 8
```
This might helps you to understand each stage's processing and directory structure.


## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `codec.sh`.

Current support models (You can refer to libritts recipe as we usually start from the corpus).
- [SoundStream](https://arxiv.org/abs/2107.03312)
- [Encodec](https://github.com/facebookresearch/encodec)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [FunCodec (Freq-Codec)](https://github.com/modelscope/FunCodec)
- [HiFiCodec](https://github.com/yangdongchao/AcademiCodec)


## FAQ

### Pre-trained codec models and usage
We provide pre-trained codec models in [ESPnet huggingface](https://huggingface.co/espnet)

A quick usage of pre-trained models is as follows:
```
from espnet2.bin.gan_codec_inference import AudioCoding
import numpy as np

# the model tag can be found in ESPnet huggingface models
codec_api = AudioCoding.from_pretrained(model_tag="espnet/libritts_soundstream16k")
audio_info = codec_api(np.zeros(16000, dtype=np.float32))
```

For advanced usage (e.g., batch tokenization and auto-packing to other tasks), see also `egs2/TEMPLATE/codec1/scripts/feats/codec_tokenization.sh`
