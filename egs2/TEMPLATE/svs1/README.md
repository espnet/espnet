# Muskits SVS Recipe TEMPLATE

This is a template of SVS recipe for Muskits.

## Table of Contents

* [Muskits SVS Recipe TEMPLATE](#muskits-svs-recipe-template)
  * [Table of Contents](#table-of-contents)
  * [Recipe flow](#recipe-flow)
    * [1\. Database-dependent data preparation](#1-database\-dependent-data-preparation)
    * [2\. Standard audio and midi formatting](#2-standard-audio-and-midi-formatting)
    * [3\. Filtering](#3-filtering)
    * [4\. Token list generation](#4-token-list-generation)
    * [5\. Statistics collection](#5-statistics-collection)
    * [6\. Model training](#6-model-training)
    * [7\. Model inference](#7-model-inference)
    * [8\. Objective evaluation](#8-objective-evaluation)
    * [9\. Model packing](#9-model-packing)
  * [How to run](#how-to-run)
    * [Multi speaker model with speaker ID embedding training](#multi-speaker-model-with-speaker-id-embedding-training)
    * [Multi language model with language ID embedding training](#multi-language-model-with-language-id-embedding-training)
    * [Vocoder training](#vocoder-training)
    * [Evaluation](#evaluation)
  * [About data directory](#about-data-directory)
  * [Supported text frontend](#supported-text-frontend)
  * [Supported text cleaner](#supported-text-cleaner)
  * [Supported Models](#supported-models)


## Recipe flow

SVS recipe consists of 8 stages.

### 1. Database-dependent data preparation

Data preparation stage starts here.
It calls `local/data.sh` to creates Kaldi-style data directories but with additional `midi.scp` and `label` in `data/` for training, validation, and evaluation sets.

See also:
- [About data directory](#about-data-directory)

### 2. Standard audio and midi formatting

If you specify `--feats_type raw` option, this is a wav dumping stage which reformats `wav.scp` in data directories.
Else, if you specify `--feats_type fbank` option or `--feats_type stft` option, this is a feature extracting stage (to be updated).
MIDI is also normalized and segmented at this stage.

Additionally, speaker ID embedding and language ID embedding preparation will be performed in this stage if you specify `--use_sid true` and `--use_lid true` options.
Note that this processing assume that `utt2spk` or `utt2lang` are correctly created in stage 1, please be careful.

### 3. Filtering

Processing stage to remove long and short utterances from the training and validation sets. 
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

Empty text will also be removed.

### 4. Token list generation

Token list generation stage.
It generates token list (dictionary) from `srctexts`.
You can change the tokenization type via `--token_type` option. `token_type=phn` are supported.
If `--cleaner` option is specified, the input text will be cleaned with the specified cleaner.
If `token_type=phn`, the input text will be converted with G2P module specified by `--g2p` option.

See also:
- [Supported text cleaner](#supported-text-cleaner).
- [Supported text frontend](#supported-text-frontend).

Data preparation will end in stage 4. You can skip data preparation (stage 1 ~ stage 4) via `--skip_data_prep` option.

### 5. Statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training and validation sets).

### 6. Model training

SVS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

Training process will end in stage 6. You can skip training process (stage 5 ~ stage 6) via `--skip_train` option.

### 7. Model inference

SVS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.
Compatible vocoder can be trained and loaded.

See also:
- [Vocoder trainging](#vocoder-training)
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 8. Objective evaluation

Evaluation stage.
It conducts four objective evaluations.
See also:
- [Evaluation](#evaluation)

### 9. Model packing

Packing stage.
It packs the trained model files.


## How to run

See [Tutorial](https://github.com/SJTMusicTeam/Muskits/blob/main/doc/tutorial.md#muskits).

As a default, we train ofuton_p_utagoe (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/           # validation set
│   ├── eval/          # evaluation set
│   ├── tr_no_dev/     # training set
│   └── token_list/    # token list (directory)
│        └── phn_none_jp/  # token list
├── dump/ # feature dump directory
│   └── raw/
│       ├── org/
│       │    ├── tr_no_dev/ # training set before filtering
│       │    └── dev/       # validation set before filtering
│       ├── srctexts   # text to create token list
│       ├── eval/      # evaluation set
│       ├── dev/       # validation set after filtering
│       └── tr_no_dev/ # training set after filtering
└── exp/ # experiment directory
    ├── svs_stats_raw_phn_none_jp  # statistics
    └── svs_train_raw_phn_none_jp  # model
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── valid/                 # valid results
        ├── decode_train.loss.best/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval/ # evaluation set
        │        ├── norm/        # generated features
        │        ├── denorm/      # generated denormalized features
        │        ├── wav/         # generated wav via Griffin-Lim
        │        ├── log/         # log directory
        │        ├── feats_type   # feature type
        │        └── speech_shape # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_2best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop-stage` options.
```sh
$ ./run.sh --stage 1 --stop-stage 1
$ ./run.sh --stage 2 --stop-stage 2
...
$ ./run.sh --stage 7 --stop-stage 7
```
This might helps you to understand each stage's processing and directory structure.


### Multi-speaker model with speaker ID embedding training

First, you need to run from the stage 2 and 3 with `--use_sid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_sid true
```
You can find the speaker ID file in `dump/raw/*/utt2sid`.
Note that you need to correctly create `utt2spk` in data prep stage to generate `utt2sid`.
Then, you can run the training with the config which has `spks: #spks` in `svs_conf`.
```yaml
# e.g.
svs_conf:
    spks: 5  # Number of speakers
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_sid true --train_config /path/to/your_multi_spk_config.yaml
```

### Multi-language model with language ID embedding training

First, you need to run from the stage 2 and 3 with `--use_lid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_lid true
```
You can find the speaker ID file in `dump/raw/*/utt2lid`.
**Note that you need to additionally create `utt2lang` file in stage 1 to generate `utt2lid`.**
Then, you can run the training with the config which has `langs: #langs` in `svs_conf`.
```yaml
# e.g.
svs_conf:
    langs: 4  # Number of languages
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --train_config /path/to/your_multi_lang_config.yaml
```

Of course you can further combine with speaker ID embedding.
If you want to use both sid and lid, the process should be like this:
```sh
$ ./run.sh --stage 2 --stop-stage 3 --use_lid true --use_sid true
```
Make your config.
```yaml
# e.g.
svs_conf:
    langs: 4   # Number of languages
    spks: 5    # Number of speakers
```
Please run the training from stage 6.
```sh
$ ./run.sh --stage 6 --use_lid true --use_sid true --train_config /path/to/your_multi_spk_multi_lang_config.yaml
```


### Vocoder training

If your `--vocoder_file` is set to none, Griffin-Lim will be used.
You can also train corresponding vocoder using [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)..

Pretrained vocoder is like follows:

```sh
*_hifigan.v1 
├── checkpoint-xxxxxxsteps.pkl
├── config.yml
└── stats.h5
```

```sh
# Use the vocoder trained by `parallel_wavegan` repo manually
$ ./run.sh --stage 7 --vocoder_file /path/to/checkpoint-xxxxxxsteps.pkl --inference_tag decode_with_my_vocoder
```

### Evaluation

We provide four objective evaluation metrics:

- Mel-cepstral distortion (MCD)
- Logarithmic rooted mean square error of the fundamental frequency (F![1](http://latex.codecogs.com/svg.latex?_0)RMSE)
- Semitone accuracy (Semitone ACC)
- Voiced / unvoiced error rate (VUV_E)

For MCD, we apply dynamic time-warping (DTW) to match the length difference between ground-truth singing and generated singing.

Here we show the example command to calculate objective metrics:

```sh
cd egs/<recipe_name>/svs1
. ./path.sh
# Evaluate MCD
./pyscripts/utils/evaluate_mcd.py \
    exp/<model_dir_name>/<decode_dir_name>/eval/wav/gen_wavdir_or_wavscp.scp \
    dump/raw/eval/gt_wavdir_or_wavscp.scp
```
```sh
cd egs/<recipe_name>/svs1
. ./path.sh
# Evaluate log-F0 RMSE & Semitone ACC & VUV Error Rate
./pyscripts/utils/evaluate_f0.py \
    exp/<model_dir_name>/<decode_dir_name>/eval/wav/gen_wavdir_or_wavscp.scp \
    dump/raw/eval/gen_wavdir_or_wavscp.scp
```
While these objective metrics can estimate the quality of synthesized singing, it is still difficult to fully determine human perceptual quality from these values, especially with high-fidelity generated singing.
Therefore, we recommend performing the subjective evaluation (eg. MOS) if you want to check perceptual quality in detail.

You can refer [this page](https://github.com/kan-bayashi/webMUSHRA/blob/master/HOW_TO_SETUP.md) to launch web-based subjective evaluation system with [webMUSHRA](https://github.com/audiolabs/webMUSHRA).

## About data directory

Each directory of training set, development set, and evaluation set, has same directory structure. See also http://kaldi-asr.org/doc/data_prep.html about Kaldi data structure. 
We recommend you running `ofuton_p_utagoe_db` recipe and checking the contents of `data/` by yourself.

```bash
cd egs/ofuton_p_utagoe_db/svs1
./run.sh
```

- Directory structure
    ```
    data/
    ├── tr_no_dev/     # Training set directory
    │   ├── text       # The transcription
    │   ├── label      # Specifying start and end time of the transcription 
    │   ├── midi.scp   # MIDI file path
    │   ├── wav.scp    # Wave file path
    │   ├── utt2spk    # A file mapping utterance-id to speaker-id
    │   ├── spk2utt    # A file mapping speaker-id to utterance-id
    │   ├── segments   # [Option] Specifying start and end time of each utterance
    │   └── (utt2lang) # A file mapping utterance-id to language type (only for multilingual)
    ├── dev/
    │   ...
    ├── test/
    │   ...
    └── token_list/   # token list directory
        ...
    ```

 - `text` format
    ```
    uttidA <transcription>
    uttidB <transcription>
    ...
    ```
    
- `label` format
    ```
    uttidA (startA1, endA1, phA1) (startA2, endA2, phA1) ...
    uttidB (startB1, endB1, phB1) (startB2, endB2, phB2) ...
    ...
    ```
    
- `midi.scp` format
    ```
    uttidA /path/to/uttidA.mid
    uttidB /path/to/uttidB.mid
    ...
    ```
    
    Note that for databases without explicit midi or MusicXML, we also provide rule-based automatic music transcription to extract related music information. Relevant functions can be found in KiSing recipe [here](https://github.com/SJTMusicTeam/Muskits/blob/main/egs/kising/svs1/local/data.sh).
    
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
    utils/utt2spk_to_spk2utt.pl data/tr_no_dev/utt2spk > data/tr_no_dev/spk2utt
    utils/spk2utt_to_utt2spk.pl data/tr_no_dev/spk2utt > data/tr_no_dev/utt2spk
    ```
    
    If your corpus doesn't include speaker information, give the same speaker id as the utterance id to satisfy the directory format, otherwise give the same speaker id for all utterances (Actually we don't use speaker information for asr recipe now).
    
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
    
- [Option] `segments` format

    If the audio data is originally long recording, about > ~1 hour, and each audio file includes multiple utterances in each section, you need to create `segments` file to specify the start time and end time of each utterance. The format is `<utterance_id> <wav_id> <start_time> <end_time>`.

    ```
    ofuton_0000000000000000hato_0007 ofuton_0000000000000000hato 33.470 38.013
    ...
    ```
    
    Note that if using `segments`, `wav.scp` has `<wav_id>` which corresponds to the `segments` instead of `utterance_id`.
    
    ```
    ofuton_0000000000000000hato /path/to/ofuton_0000000000000000hato.wav
    ...
    ```
    
- `utt2lang` format
    ```
    uttidA languageA
    uttidB languageB
    uttidC languageA
    uttidD lagnuageB
    ...
    ```
    
    Note that `utt2lang` file is only generated for multilingual dataset (see in recipe `egs/multilingual_four`).
    
Once you complete creating the data directory, it's better to check it by `utils/validate_data_dir.sh`.

```bash
utils/validate_data_dir.sh --no-feats data/tr_no_dev
utils/validate_data_dir.sh --no-feats data/dev
utils/validate_data_dir.sh --no-feats data/test
```


## Supported text frontend

You can change via `--g2p` option in `svs.sh`.

- `none`: Just separate by space
    - e.g.: `HH AH0 L OW1 <space> W ER1 L D` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`

You can see the code example from [here](https://github.com/SJTMusicTeam/Muskits/blob/main/muskit/text/phoneme_tokenizer.py).


## Supported text cleaner

You can change via `--cleaner` option in `svs.sh`.

- `none`: No text cleaner.

You can see the code example from [here](https://github.com/SJTMusicTeam/Muskits/blob/main/muskit/text/cleaner.py).

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `run.sh`.

- [Naive-RNN](https://arxiv.org/abs/2010.12024)
- [GLU-Transformer](https://arxiv.org/abs/1910.09989)
- [MLP-Singer](https://arxiv.org/abs/2106.07886)
- [XiaoIce](https://arxiv.org/pdf/2006.06261)

You can find example configs of the above models in [`egs/ofuton_p_utagoe_db/svs1/conf/tuning`](../../ofuton_p_utagoe_db/svs1/conf/tuning).



