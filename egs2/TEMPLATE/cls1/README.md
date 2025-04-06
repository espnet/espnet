# Classification

This is a template of cls1 recipe for ESPnet2.

## Table of Contents

- [Recipe flow](#recipe-flow)
    - [1. Database-dependent data preparation](#1-database-dependent-data-preparation)
    - [2. Wav dump preparation](#2-wav-dump-preparation)
    - [3. Filtering](#3-filtering)
    - [4. Token list generation](#4-token-list-generation)
    - [5. CLS statistics collection](#5-cls-statistics-collection)
    - [6. CLS training](#6-cls-training)
    - [7. CLS inference](#7-cls-inference)
    - [8. Scoring](#8-scoring)
    - [9. Model packing](#9-model-packing)
    - [10. Upload to HuggingFace](#9-model-upload)
- [How to run](#how-to-run)
- [Evaluation](#evaluation)
- [About data directory](#about-data-directory)
- [Problems you might encounter](#problems-you-might-encounter)
    - [1. Torcheval not found](#1-torcheval-not-found)
- [Supported Models](#supported-models)


## Recipe flow

CLS recipe consists of 10 stages.

### 1. Database-dependent data preparation

Data preparation stage.
It calls `local/data.sh` to creates Kaldi-style data directories for training, validation, and evaluation sets.

See also:
- [About data directory](#about-data-directory)
- [Score preparation](#score-preparation)

### 2. Wav dump preparation

This recipe supports `--feats_type raw` option.
This means we will run a wav dumping stage which reformats `wav.scp` in data directories.
This process standardizes all data to common sampling rate and data format.

### 3. Filtering

Filtering stage.
Processing stage to remove long and short utterances from the training and validation sets.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

Empty text will also be removed.
If your audio sample lacks a label in multi-label setting then use the `<blank>` symbol.
TODO(shikhar): This feature will be supported in a later PR.

### 4. Token list generation

Token list generation stage.
It generates token list (dictionary) from `text` file.
We only support `--token_type=word` option.
This means that each unique space-separated word in the text file becomes a class/label for classification.
Please note that this process is case-sensitive.


NOTE: Data preparation will end in stage 4. You can skip data preparation (stage 1 ~ stage 4) via `--skip_data_prep` option.

### 5. CLS statistics collection

Statistics calculation stage. It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training and validation sets).

### 6. CLS training

Classification model training stage.
You can change the training setting via `--train_config` and `--cls_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

Training process will end in stage 6. You can skip training process (stage 5 ~ stage 6) via `--skip_train` option.

### 7. CLS inference

Classification model decoding stage.
This stage outputs two files: text and score.

```
Example text file

as20k-eval-0 Music
as20k-eval-1
```
For multi-class classification each row will have exactly one class.
For multi-label classification each row can have any number of labels (zero or more).
The above example is a multi-label output text file.

```
Example score file

as20k-eval-0 0.5590277314186096 0.451458394527435 ...
as20k-eval-1 0.00023992260685190558 0.00012396479723975062 ...
```
Each row of both multi-class and multi-label classification models will have probabilities for all tokens (in the same order as they are present in the `token_list`).

We use a threshold of 0.5 for multi-label classification, and use argmax for multi-class classification.
You can choose to just produce probabilities for the predicted class/labels in the score file with `output_all_probabilities=false` flag.

### 8. Scoring

Evaluation stage.
It produces mAP and accuracy metrics.

### 9. Model packing

Packing stage.
It packs the trained model files.
Set `skip_upload` to `False`.

### 10. Model upload

Upload stage.
It uploads the trained model files.
Provide `hf_repo` and set `skip_upload` to `False`.

## How to run
TOOD(shikhar): Change this to a recipe which downloads data (perhaps beans) later.

Here, we show the procedure to run the recipe using `egs2/as20k/cls1`.

Move on the recipe directory.
```sh
$ cd egs2/as20k/cls1
```

Modify `AUDIOSET` variable in `db.sh` to specify location where you have the AudioSet dataset.
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

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop_stage` options.
```sh
$ ./run.sh --stage 1 --stop_stage 1
$ ./run.sh --stage 2 --stop_stage 2
...
$ ./run.sh --stage 7 --stop_stage 7
```
This might help you understand each stage's processing and directory structure.

### Evaluation

Here we show the example command to calculate classification metrics:

```sh

cd egs2/<recipe_name>/cls1
. ./path.sh

python3 pyscripts/utils/cls_score.py \
    -gtxt data/text \
    -ptxt exp/cls_<split>/text \
    -pscore exp/cls_<split>/score \
    -tok data/token_list

```

## About data directory

Each directory of training set, development set, and evaluation set, has same directory structure. See also https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory about Kaldi data structure.

- Directory structure
    ```
    data/
    ├── train/     # Training set directory
    │   ├── text       # The transcription
    │   ├── wav.scp    # Wave file path
    │   ├── utt2spk    # A file mapping utterance-id to speaker-id
    │   ├── spk2utt    # A file mapping speaker-id to utterance-id
    |
    ├── dev/
    │   ...
    ├── eval/
    │   ...
    └── token_list   # token list file
        ...
    ```

 - `text` format
    ```
    uttidA <class_a>
    uttidB <class_b1> <class_b2>
    ...
    ```
    Note that for multi-class classification each uttid should be associated with exactly one class.
    For multi-label classification, each uttid should have at least one label.
    (TODO) We will support the case with no label in the future with the `<blank>` symbol.

- `wav.scp` format
    ```
    uttidA /path/to/uttidA.wav
    uttidB /path/to/uttidB.wav
    ...
    ```

- `utt2spk` format
    ```
    uttidA speakerA
    uttidB speakerB
    uttidC speakerA
    uttidD speakerB
    ...
    ```

- `spk2utt` format
    ```
    speakerA uttidA uttidC ...
    speakerB uttidB uttidD ...
    ...
    ```

    Note that `spk2utt` file can be generated by `utt2spk`, and `utt2spk` can be generated by `spk2utt`, so it's enough to create either one of them.

    ```bash
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
    utils/spk2utt_to_utt2spk.pl data/train/spk2utt > data/train/utt2spk
    ```

    If your corpus doesn't include speaker information, give the same speaker id as the utterance id to satisfy the directory format, otherwise give the same speaker id for all utterances (Actually we don't use speaker information for cls1 recipe now).

    ```bash
    uttidA uttidA
    uttidB uttidB
    ...
    ```

    OR

    ```bash
    uttidA dummy
    uttidB dummy
    ...
    ```

Once you complete creating the data directory, it's good to check it by `utils/validate_data_dir.sh`.

```bash
utils/validate_data_dir.sh --no-feats data/train
utils/validate_data_dir.sh --no-feats data/dev
utils/validate_data_dir.sh --no-feats data/test
```

### Problems you might encounter

Below are some common errors to watch out for:

#### 1. Torcheval not found
* Run `pip install torcheval`


## Supported Models
TODO(shikhar): Add details about BEATs once it is trained.
