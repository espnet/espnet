# ESPnet2 Spk1 Recipe TEMPLATE

This is a template of Spk1 recipe for ESPnet2.
It follows d-vector style training/inference for speaker verification.
In other words, it trains a DNN as a closed set speaker classifier.
After training the classification head is removed. The last hidden layer
(or sometimes another layer) is used as a speaker representation (i.e.,
speaker embedding) to represent diverse open set speakers.

## Table of Contents

* [ESPnet2 SPK1 Recipe TEMPLATE](#ESPnet2-Spk1-Recipe-TEMPLATE)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Data preparation](#1-data-preparation)
    * [2\. Speed perturbation](#2-speed-perturbation)
    * [3\. Wav format](#3-wav-format)
    * [4\. Spk statistics collection](#4-spk-statistics-collection)
    * [5\. Spk training](#5-spk-training)
    * [6\. Speaker embedding extraction](#6-speaker-embedding-extraction)
    * [7\. Score calculation](#7-score-calculation)
    * [8\. Metric calculation](#8-metric-calculation)
    * [9\-10\. (Optional) Pack results for upload](#9-10-optional-pack-results-for-upload)
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

### 2. Speed perturbation
Generate train data with different speed offline, as a form of augmentation.

### 3. Wav format

Format the wave files in `wav.scp` to a single format (wav / flac / kaldi_ark).

### 4. Spk statistics collection

Statistics calculation stage.
It collects the shape information of input and output texts for Spk training.
Currently, it's close to a dummy because we set all utterances to have equal
duration in the training phase.

### 5. Spk training

Spk model training stage.
You can change the training setting via `--spk_config` and `--spk_args` options.

See also:
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

### 6. Speaker embedding extraction
Extracts speaker embeddings for inference.
Speaker embeddings belonging to the evaluation set are extracted.
If `score_norm=true` and/or `qmf_func=true`, cohort set(s) for score normalization and/or quality measure function is also extracted.

### 7. Score calculation
Calculates speaker similarity scores for an evaluation protocol (i.e., a set of trials).
One scalar score is calcuated for each trial.

This stage includes score normalization if set with `--score_norm=true`.
This stage includes score normalization if set with `--qmf_func=true`.

### 8. Metric calculation
Calculates equal error rates (EERs) and minimum detection cost function (minDCF).

### 9-10. (Optional) Pack results for upload

Packing stage.
It packs the trained model files and uploads to Huggingface.
If you want to run this stage, you need to register your account in Huggingface.

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
