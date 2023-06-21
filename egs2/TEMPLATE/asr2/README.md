# ESPnet2 ASR2 Recipe TEMPLATE

This is a template of ASR2 recipe for ESPnet2.
The difference from ASR1 is that discrete tokens are used as input instead of conventional audios / spectrum features.

## Table of Contents

* [ESPnet2 ASR2 Recipe TEMPLATE](#espnet2-asr2-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Speed perturbation](#2-speed-perturbation)
    * [3\. Wav format](#3-wav-format)
    * [4\. Generate discrete tokens](#4-generate-discrete-tokens)
    * [5\. Generate dump folder](#5-generate-dump-folder)
    * [6\. Removal of long / short data](#6-removal-of-long--short-data)
    * [7\. Input / Output Token list generation](#7-input-output-token-list-generation)
    * [8\. LM statistics collection](#8-lm-statistics-collection)
    * [9\. LM training](#9-lm-training)
    * [10\. LM perplexity](#10-lm-perplexity)
    * [11\. Ngram-LM training](#11-ngram-lm-training)
    * [12\. ASR statistics collection](#12-asr-statistics-collection)
    * [13\. ASR training](#13-asr-training)
    * [14\. ASR inference](#14-asr-inference)
    * [15\. ASR scoring](#15-asr-scoring)
    * [16\-18\. (Optional) Pack results for upload](#16-18-optional-pack-results-for-upload)
  * [How to run](#how-to-run)
    * [LibriSpeech training](#librispeech-training)
  * [Related works](#related-works)

## Recipe flow

ASR2 recipe consists of 15 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to creates Kaldi-style data directories in `data/` for training, validation, and evaluation sets. It's the same as `asr1` tasks.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Speed perturbation

Augment training data with speed perturbation. `data/${train_set}_spXX` would be generated (`XX` means the speed factor). This step is optional.

### 3. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 4. Generate discrete tokens

The discrete tokens of the input speech signals are generated. For ASR2 task, the input is the discrete tokens (from self-supervised learning (SSL) features) and the target is the ASR transcriptions. After getting the discrete tokens (usually in integers), they will be converted to CJK characters, which are more convenient in tokenization.
#### Input / Target / Process of data preparation

- Stages:
  1. Generate SSL features for train / valid / test sets.
  2. Train the K-Means model on a subset from training data.
  3. Generate K-Means-based discrete tokens for train / valid / test sets.
  4. (Optional) Measure the discrete tokens quality if forced-alignment is accessible.


### 5. Generate dump folder

Dumping stage.
This stage move necessary files for training from `data` folder to `dump` folder.

### 6. Removal of long / short data

TODO.

### 7. Token list generation

Token list (BPE / Char / etc) generation for both input and targets.

### 8. LM statistics collection

Neural-network (NN) based Language model (LM) is optional for ASR task. You can skip stage 5-8 by setting `--use_lm false`.
Statistics calculation stage.
It collects the shape information of LM texts and calculates statistics for LM training.

### 9. LM training

NN-based LM model training stage.
You can change the training setting via `--lm_config` and `--lm_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 10. LM perplexity

NN-based LM evaluation stage. Perplexity (PPL) is computed against the trained model

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 11. N-gram LM training

N-gram-based LM model training stage.


### 12. ASR statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for ASR training.

### 13. ASR training

ASR model training stage.
You can change the training setting via `--asr_config` and `--asr_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 14. ASR inference

ASR inference stage.

### 15. ASR scoring

ASR scoring stage: error rates (char / word / token) are computed.

### 16-18. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to [Zenodo](https://zenodo.org/) (Zenodo upload will be deprecated).
If you want to run this stage, you need to register your account in zenodo.

See also:
- [ESPnet Model Zoo](https://github.com/espnet/espnet_model_zoo)

#### Stage 16-18: Upload model

Upload the trained model to Hugging Face for sharing. Additional information at [Docs](https://espnet.github.io/espnet/espnet2_tutorial.html#packing-and-sharing-your-trained-model).

## How to run

### LibriSpeech Training
Here, we show the procedure to run the recipe using `egs2/librispeech/asr2`.

Move on the recipe directory.
```sh
$ cd egs2/librispeech/asr2
```

Modify `LIBRISPEECH` variable in `db.sh` if you want to change the download directory.
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

## Related works
```
@INPROCEEDINGS{9054224,
  author={Baevski, Alexei and Mohamed, Abdelrahman},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Effectiveness of Self-Supervised Pre-Training for ASR},
  year={2020},
  volume={},
  number={},
  pages={7694-7698},
  doi={10.1109/ICASSP40776.2020.9054224}}

@article{chang2023exploration,
  title={Exploration of Efficient End-to-End ASR using Discretized Input from Self-Supervised Learning},
  author={Chang, Xuankai and Yan, Brian and Fujita, Yuya and Maekaku, Takashi and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2305.18108},
  year={2023}
}
```
