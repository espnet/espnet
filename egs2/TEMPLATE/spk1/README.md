# ESPnet2 Spk1 Recipe TEMPLATE

This is a template of Spk1 recipe for ESPnet2.
It follows d-vector style training/inference for speaker verification.
In other words, it trains a DNN as a closed set speaker classifier.
After training the classification head is removed. The last hidden layer
(or sometimes another layer) is used as a speaker representation (i.e.,
speaker embedding) to represent diverse open set speakers.

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

Spk1 recipe consists of 4 stages.

### 1. Data preparation

Data preparation stage.

#### ESPnet format:

It calls `local/data.sh` to create Kaldi-style data directories in `data/` for training, validation, and evaluation sets. It's the same as `asr1` tasks.

See also:
- [About Kaldi-style data directory](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory)

### 2. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 3. Spk statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for Spk training.
Currently, it's close to a dummy because we set all utterances to have equal
duration in the training phase.

### 4. Spk training

Spk model training stage.
You can change the training setting via `--spk_config` and `--spk_args` options.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

## How to run

### VoxCeleb Training
Here, we show the procedure to run the recipe using `egs2/voxceleb/spk1`.

Move to the recipe directory.
```sh
$ cd egs2/voxceleb/spk1
```

Modify `VOXCELEB1`, `VOXCELEB2` variables in `db.sh` if you want to change the download directory.
```sh
$ vim db.sh
```

Modify `cmd.sh` and `conf/*.conf` if you want to use the job scheduler.
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
@INPROCEEDINGS{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  year={2022},
  booktitle={Proc. INTERSPEECH}
}
```
