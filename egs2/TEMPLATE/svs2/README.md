# ESPnet2 SVS2 Recipe TEMPLATE

This is a template of SVS2 recipe for ESPnet2.

SVS2 replace intermediate features in SVS1 (e.g. mel-spectrograms or continuous features) with discrete tokens, which are usually extracted from pretrained self-supervised learning (SSL) models or codec based models.

## Table of Contents

- [ESPnet2 SVS2 Recipe TEMPLATE](#espnet2-svs2-recipe-template)
  - [Table of Contents](#table-of-contents)
  - [Recipe flow](#recipe-flow)
    - [1. Database-dependent data preparation](#1-database-dependent-data-preparation)
    - [2. Wav dump / Embedding preparation](#2-wav-dump--embedding-preparation)
    - [3. Filtering](#3-filtering)
    - [4. Discrete token extraction using pretrained model](#4-discrete-token-extraction-using-pretrained-model)
    - [5. Token list generation](#5-token-list-generation)
    - [6. SVS statistics collection](#6-svs-statistics-collection)
    - [7. SVS training](#7-svs-training)
    - [8. SVS inference](#8-svs-inference)
    - [9. Objective evaluation](#9-objective-evaluation)
    - [10. Model packing](#10-model-packing)
  - [How to run](#how-to-run)
    - [Naive\_RNN\_DP training](#naive_rnn_dp-training)
    - [XiaoiceSing training](#xiaoicesing-training)
    - [TokSing training](#toksing-training)
    - [Multi-speaker model with speaker ID embedding training](#multi-speaker-model-with-speaker-id-embedding-training)
    - [Multi-language model with language ID embedding training](#multi-language-model-with-language-id-embedding-training)
    - [Vocoder training](#vocoder-training)
    - [Evaluation](#evaluation)
  - [About data directory](#about-data-directory)
  - [Score preparation](#score-preparation)
      - [Case 1: phoneme annotation and standardized score](#case-1-phoneme-annotation-and-standardized-score)
      - [Case 2: phoneme annotation only](#case-2-phoneme-annotation-only)
    - [Problems you might meet](#problems-you-might-meet)
      - [1. Wrong segmentation point](#1-wrong-segmentation-point)
      - [2. Wrong lyric / midi annotation](#2-wrong-lyric--midi-annotation)
      - [3. Different lyric-phoneme pairs against the given g2p](#3-different-lyric-phoneme-pairs-against-the-given-g2p)
      - [4. Special marks in MusicXML](#4-special-marks-in-musicxml)
  - [Supported text cleaner](#supported-text-cleaner)
  - [Supported text frontend](#supported-text-frontend)
  - [Supported Pretrained Model](#supported-pretrained-model)
  - [Supported Models](#supported-models)


## Recipe flow

SVS recipe consists of 9 stages.

### 1. Database-dependent data preparation

Data preparation stage.

It calls `local/data.sh` to creates Kaldi-style data directories but with additional `score.scp` and `label` in `data/` for training, validation, and evaluation sets.

See also:
- [About data directory](#about-data-directory)
- [Score preparation](#score-preparation)

### 2. Wav dump / Embedding preparation

If you specify `--feats_type raw` option, this is a wav dumping stage which reformats `wav.scp` in data directories.

Else, if you specify `--feats_type fbank` option or `--feats_type stft` option, this is a feature extracting stage (to be updated).

Additionally, speaker ID embedding and language ID embedding preparation will be performed in this stage if you specify `--use_sid true` and `--use_lid true` options.
Note that this processing assume that `utt2spk` or `utt2lang` are correctly created in stage 1, please be careful.

### 3. Filtering

Filtering stage.

Processing stage to remove long and short utterances from the training and validation sets.
You can change the threshold values via `--min_wav_duration` and `--max_wav_duration`.

Empty text will also be removed.


### 4. Discrete token extraction using pretrained model
Discrete token extraction stage.

Two types of discrete token are supported in SVS2: *Single layer token* and *Multi layer token*.

*Single layer token* will only extract discrete token from single layer of the pretrained model.
The RVQ extraction is also available and the follwing application of RVQ discrete tokens will be same as *Multi layer token*.

*Multi layer token* will combine different layer discrete token from *Single layer token* or *RVQ* together in `frame` or `sequence` ways.

More details can ben found in [Data Preparation](#data-preparation).

See also:
- [Supported pretrained model](#supported-pretrained-model)


### 5. Token list generation

Token list generation stage.

It generates token list (dictionary) from `srctexts`.
You can change the tokenization type via `--token_type` option. `token_type=phn` are supported.
If `--cleaner` option is specified, the input text will be cleaned with the specified cleaner.
If `token_type=phn`, the input text will be converted with G2P module specified by `--g2p` option.

See also:
- [Supported text cleaner](#supported-text-cleaner).
- [Supported text frontend](#supported-text-frontend).

Data preparation will end in stage 4. You can skip data preparation (stage 1 ~ stage 4) via `--skip_data_prep` option.

### 6. SVS statistics collection

Statistics calculation stage.
It collects the shape information of the input and output and calculates statistics for feature normalization (mean and variance over training and validation sets).

In this stage, you can set `--write_collected_feats true` to store statistics of pitch and feats.

### 7. SVS training

SVS model training stage.
You can change the training setting via `--train_config` and `--train_args` options.

See also:
- [Supported models](#supported-models).
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)
- [Distributed training](https://espnet.github.io/espnet/espnet2_distributed.html)

Training process will end in stage 6. You can skip training process (stage 5 ~ stage 6) via `--skip_train` option.

### 8. SVS inference

SVS model decoding stage.
You can change the decoding setting via `--inference_config` and `--inference_args`.
Compatible vocoder can be trained and loaded.

See also:
- [Vocoder training](#vocoder-training)
- [Change the configuration for training](https://espnet.github.io/espnet/espnet2_training_option.html)

### 9. Objective evaluation

Evaluation stage.
It conducts four objective evaluations.
See also:
- [Evaluation](#evaluation)

### 10. Model packing

Packing stage.
It packs the trained model files.


## How to run

Here, we show the procedure to run the recipe using `egs2/opencpop/svs2`.

Move on the recipe directory.
```sh
$ cd egs2/opencpop/svs2
```

Modify `OPENCPOP` variable in `db.sh` if you want to change the download directory.
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
As a default, we train Naive_RNN (`conf/train.yaml`) with `feats_type=raw` + `token_type=phn`.

Then, you can get the following directories in the recipe directory.
```sh
├── data/ # Kaldi-style data directory
│   ├── dev/           # validation set
│   ├── eval/          # evaluation set
│   ├── tr_no_dev/     # training set
│   └── token_list/    # token list (directory)
│        └── phn_none_zh/  # token list
├── dump/ # feature dump directory
│   ├── raw/
│   │   ├── org/
│   │   │    ├── tr_no_dev/ # training set before filtering
│   │   │    └── dev/       # validation set before filtering
│   │   ├── srctexts   # text to create token list
│   │   ├── eval/      # evaluation set
│   │   ├── dev/       # validation set after filtering
│   │   └── tr_no_dev/ # training set after filtering
│   └── extracted/ # log direcotry for feature extraction 
│       └── [model_name]/ # pretrained model workspace
│           └── [layer_index]/ # specific layer for pretrained model
│               ├── eval/      
│               ├── dev/       
│               ├── tr_no_dev/ 
│               └── tr_no_dev_subset[portion]/ # subset of train set
└── exp/ # experiment directory
    ├── svs_stats_raw_phn_none_jp  # statistics
    │   ├── logdir/ # statistics calculation log directory
    │   ├── train/  # train statistics
    │   ├── valid/  # valid statistics
    └── svs_train_raw_phn_none_jp  # model
        ├── tensorboard/           # tensorboard log
        ├── images/                # plot of training curves
        ├── valid/                 # valid results
        ├── decode_train.loss.best/ # decoded results
        │    ├── dev/   # validation set
        │    └── eval/ # evaluation set
        │        ├── norm/         # generated features
        │        ├── denorm/       # generated denormalized features
        │        ├── MCD_res/      # mel-cepstral distortion
        │        ├── VUV_res/      # voiced/unvoiced error rate
        │        ├── SEMITONE_res/ # semitone accuracy
        │        ├── F0_res/       # log-F0 RMSE
        │        ├── wav/          # generated wav via vocoder
        │        ├── log/          # log directory
        │        ├── feats_type    # feature type
        │        └── speech_shape  # shape info of generated features
        ├── config.yaml             # config used for the training
        ├── train.log               # training log
        ├── *epoch.pth              # model parameter file
        ├── checkpoint.pth          # model + optimizer + scheduler parameter file
        ├── latest.pth              # symlink to latest model parameter
        ├── *.ave_2best.pth         # model averaged parameters
        └── *.best.pth              # symlink to the best model parameter loss
```

In decoding, you can see [vocoder training](#vocoder-training) to set vocoder.

For the first time, we recommend performing each stage step-by-step via `--stage` and `--stop_stage` options.
```sh
$ ./run.sh --stage 1 --stop_stage 1
$ ./run.sh --stage 2 --stop_stage 2
...
$ ./run.sh --stage 8 --stop_stage 8
```
This might help you understand each stage's processing and directory structure.

### Data Preparation 

First, ensure that the settings in `run.sh` such as `fs` and `n_shift` match those in pretrained models (use 6th layer of model *wavlm_large* as the exmpale).

Complete the dataset preprocess:
```sh
$ ./run.sh \
    --stage 1 \
    --stop_stage 3 \
    --fs 16000 \
    --n_shift 320 
```

**NOTICE: This setting is mainly for self-supervised learning (SSL) model.**

Then, set the configuration for discrete token, including `nclusters`, `RVQ_layers` and `kmeans_feature`.
If *multi layer token* is chosen or `RVQ_layer` is greater than 1, configurations `multi_token` and `mix_type` should also be specified.

#### Single layer token
In *Single layer token*, it extracts discrete tokens from single layer feature of pretrained model and RVQ exctraction is also available.

In this setting, `kmeans_feature` is in the format `[model]/[layer index]`, where `[model]` shows the pretrained model and `[layer index]` represents the layer number in pretrained model.

For exmpale, if you want to get discrete tokens from 3rd layer of HuBERT base model, you can set `kmeans_feature="hubert_base/3"` and `RVQ_layers=1`. `nclusters` is usually set at `128` or `1024`.

If `RVQ_layers` is greater than 1, discret token are extracted in residual vector quantization way, which will generate `RVQ_layers` layers of discrete tokens finally.
The following application of RVQ discrete tokens is used in *Multi layer token*.

#### Multi layer token
*Multi layer token* will combine multi layers of discrete tokens extracted in *Single layer token* or *RVQ*.
Therefore, to utilize this type of tokens, you should get corresponding discrete tokens in *Single layer token* at first.

In this setting, `kmeans_feature` should be specified in the format `multi/[tag]` where `[tag]` is the new name for combined tokens. 
Discrete tokens in all layers should be listed in `multi_token`, seprated by a blank space.

For exmpale, if you want to combine discrete tokens from 6th layer of hubert large, 6th layer and 23rd layer of wavlm large, you should set `multi_token="hubert_large_ll60k_128_6_RVQ_0 wavlm_large_128_6_RVQ_0 wavlm_large_128_23_RVQ_0"` with the setting `RVQ_layers=2`. 
If the setting is `RVQ_layers=1`, the configuration will be `multi_token="hubert_large_ll60k_128_6 wavlm_large_128_6 wavlm_large_128_23"`.

When input these token into SVS2 model, it should be mix in a single token sequence. You can set `mix_type="frame"` or `mix_type="sequence"` to get the single token sequence.

Exmpale for `mix_type`: 
```py
l_A=[1, 2, 3]
l_B=[a, b, c].
# mix l_A and l_B
"frame": [1, a, 2, b, 3, c] # frame by frame
"sequence": [1, 2, 3, a, b, c] # sequence by sequence
```

Extract discrete token:
```sh
# Single layer token (RVQ_layers=1)
$ ./run.sh \
    --stage 4 \
    --stop_stage 4 \
    --fs 16000 \
    --n_shift 320 \
    --nclusters 128 \
    --RVQ_layers 1 \
    --kmeans_feature wavlm_large/6

$ ./run.sh \
    --stage 4 \
    --stop_stage 4 \
    --fs 16000 \
    --n_shift 320 \
    --nclusters 128 \
    --RVQ_layers 1 \
    --kmeans_feature wavlm_large/23

# Multi layer token(RVQ_layers=1)
$ ./run.sh \
    --stage 4 \
    --stop_stage 4 \
    --fs 16000 \
    --n_shift 320 \
    --nclusters 128 \
    --RVQ_layers 1 \
    --mix_type frame \
    --kmeans_feature multi/wavlm_large_6+wavlm_large_23 \
    --multi_token "wavlm_large_128_6 wavlm_large_128_23"

# Single layer token (RVQ_layers>1)
$ ./run.sh \
    --stage 4 \
    --stop_stage 4 \
    --fs 16000 \
    --n_shift 320 \
    --nclusters 128 \
    --RVQ_layers 3 \
    --kmeans_feature wavlm_large/6

# Multi layer token(RVQ_layers>1)
$ ./run.sh \
    --stage 4 \
    --stop_stage 4 \
    --fs 16000 \
    --n_shift 320 \
    --nclusters 128 \
    --RVQ_layers 3 \
    --mix_type frame \
    --kmeans_feature multi/wavlm_large_6_RVQ_0+1+2 \
    --multi_token "wavlm_large_128_6_RVQ_0 wavlm_large_128_6_RVQ_1 wavlm_large_128_6_RVQ_2"
```


### Naive_RNN_DP training
First, complete the [data preparation](#data-preparation).

Second, check "train_config" (default `conf/train.yaml`), "score_feats_extract" (*syllable level* in RNN_DP) and modify "vocoder_file" with your own vocoder path.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_naive_rnn_dp.yaml \
    --score_feats_extract syllable_score_feats \
    --pitch_extract dio \
    --vocoder_file ${your vocoder path} \
    ${command in data preparation}
```

### XiaoiceSing training
First, complete the [data preparation](#data-preparation).

Second, check "train_config" (default `conf/train.yaml`), "score_feats_extract" (*syllable level* in XiaoiceSing) and modify "vocoder_file" with your own vocoder path.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_xiaoice.yaml \
    --score_feats_extract syllable_score_feats \
    --pitch_extract dio \
    --vocoder_file ${your vocoder path} \
    ${command in data preparation}
```
### TokSing training
First, complete the [data preparation](#data-preparation).

Second, check "train_config" (default `conf/train.yaml`), "score_feats_extract" (*syllable level* in XiaoiceSing) and modify "vocoder_file" with your own vocoder path.
```sh
$ ./run.sh --stage 6 \
    --train_config conf/tuning/train_discrete_acoustic.yaml \
    --score_feats_extract syllable_score_feats \
    --pitch_extract dio \
    --vocoder_file ${your vocoder path} \
    ${command in data preparation}
```

### Multi-speaker model with speaker ID embedding training

First, you need to run from the stage 2 and 3 with `--use_sid true` to extract speaker ID.
```sh
$ ./run.sh --stage 2 --stop_stage 3 --use_sid true
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
$ ./run.sh --stage 2 --stop_stage 3 --use_lid true
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
$ ./run.sh --stage 2 --stop_stage 3 --use_lid true --use_sid true
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
You can also train corresponding vocoder using [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN).

Pretrained vocoder is like follows:

```sh
*_hifigan.v1
├── checkpoint-xxxxxxsteps.pkl
├── config.yml
└── stats.h5
```

```sh
# Use the vocoder trained by `parallel_wavegan` repo manually
$ ./run.sh --stage 8 --vocoder_file /path/to/checkpoint-xxxxxxsteps.pkl --inference_tag decode_with_my_vocoder
```

### Evaluation

We provide four objective evaluation metrics:

- Mel-cepstral distortion (MCD)
- Logarithmic rooted mean square error of the fundamental frequency (log-F0 RMSE)
- Semitone accuracy (Semitone ACC)
- Voiced / unvoiced error rate (VUV_E)
- Word/character error rate (WER/CER, optional executated by users)
- pseudo MOS (coming soon)

For MCD, we apply dynamic time-warping (DTW) to match the length difference between ground-truth singing and generated singing.

Here we show the example command to calculate objective metrics:

```sh
cd egs2/<recipe_name>/svs1
. ./path.sh
# Evaluate MCD & log-F0 RMSE & Semitone ACC & VUV Error Rate
./pyscripts/utils/evaluate_*.py \
    exp/<model_dir_name>/<decode_dir_name>/eval/wav/gen_wavdir_or_wavscp.scp \
    dump/raw/eval/gt_wavdir_or_wavscp.scp

# Evaluate CER
./scripts/utils/evaluate_asr.sh \
    --model_tag <asr_model_tag> \
    --nj 1 \
    --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
    --gt_text "dump/raw/eval1/text" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results

# Since ASR model does not use punctuation, it is better to remove punctuations if it contains
./scripts/utils/remove_punctuation.pl < dump/raw/eval1/text > dump/raw/eval1/text.no_punc
./scripts/utils/evaluate_asr.sh \
    --model_tag <asr_model_tag> \
    --nj 1 \
    --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
    --gt_text "dump/raw/eval1/text.no_punc" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results

# You can also use openai whisper for evaluation
./scripts/utils/evaluate_asr.sh \
    --whisper_tag base \
    --nj 1 \
    --gt_text "dump/raw/eval1/text" \
    exp/<model_dir_name>/<decode_dir_name>/eval1/wav/wav.scp \
    exp/<model_dir_name>/<decode_dir_name>/asr_results
```

While these objective metrics can estimate the quality of synthesized singing, it is still difficult to fully determine human perceptual quality from these values, especially with high-fidelity generated singing.
Therefore, we recommend performing the subjective evaluation (eg. MOS) if you want to check perceptual quality in detail.

You can refer [this page](https://github.com/kan-bayashi/webMUSHRA/blob/master/HOW_TO_SETUP.md) to launch web-based subjective evaluation system with [webMUSHRA](https://github.com/audiolabs/webMUSHRA).

## About data directory

Each directory of training set, development set, and evaluation set, has same directory structure. See also https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE#about-kaldi-style-data-directory about Kaldi data structure.
We recommend you running `opencpop` recipe and checking the contents of `data/` by yourself.

```bash
cd egs/opencpop/svs2
./run.sh
```

- Directory structure
    ```
    data/
    ├── tr_no_dev/     # Training set directory
    │   ├── text       # The transcription
    │   ├── label      # Specifying start and end time of the transcription
    │   ├── score.scp  # Score file path
    │   ├── wav.scp    # Wave file path
    │   ├── utt2spk    # A file mapping utterance-id to speaker-id
    │   ├── spk2utt    # A file mapping speaker-id to utterance-id
    │   ├── segments   # [Option] Specifying start and end time of each utterance
    │   └── (utt2lang) # A file mapping utterance-id to language type (only for multilingual)
    ├── dev/
    │   ...
    ├── eval/
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

- `score.scp` format
    ```
    key1 /some/path/score.json
    key2 /some/path/score.json
    ...
    ```

    Note that for databases without explicit score or MusicXML, we will provide rule-based automatic music transcription to extract related music information in the future.

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

## Score preparation

To prepare a new recipe, we first split songs into segments via `--silence` option if no official segmentation provided.

Then, we transfer the raw data into `score.json`, where situations can be categorized into two cases depending on the annotation:

#### Case 1: phoneme annotation and standardized score

- If the phonemes and notes are aligned in time domain, convert the raw data directly. (eg. [Opencpop](https://github.com/espnet/espnet/tree/master/egs2/opencpop/svs1))

- If the phoneme annotation are misaligned with notes in time domain, align phonemes (from `label`) and note-lyric pairs (from `musicXML`) through g2p. (eg. [Ofuton](https://github.com/espnet/espnet/tree/master/egs2/ofuton_p_utagoe_db/svs1))

- We also offer some automatic fixes for missing silences in the dataset. During the stage1, when you encounter errors such as "Lyrics are longer than phones" or "Phones are longer than lyrics", the scripts will auto-generated the fixing code. You may need to put the code into the `get_error_dict` method in `egs2/[dataset name]/svs1/local/prep_segments.py`. Noted that depending on the suggested input_type, you may want to copy it into either the `hts` or `xml`'s error_dict. (For more information, please check [namine](https://github.com/espnet/espnet/tree/master/egs2/namine_ritsu_utagoe_db/svs1) or [natsume](https://github.com/espnet/espnet/tree/master/egs2/natsume/svs1)

Specially, the note-lyric pairs can be rebuilt through other melody files, like `MIDI`, if there's something wrong with the note duration. (eg. [Natsume](https://github.com/espnet/espnet/tree/master/egs2/natsume/svs1))


#### Case 2: phoneme annotation only

 To be updated.

### Problems you might meet

During stage 1, which involves data preparation, you may encounter `ValueError` problems that typically indicate errors in the annotation. To address these issues, it is necessary to manually review the raw data in the corresponding sections and make the necessary corrections. While other toolkits and open-source codebases may not impose such requirements or checks, we have found that investing time to resolve these errors significantly enhances the quality of the singing voice synthesizer.

Note that modifications can be made to the raw data locally or through the processing data flow at stage 1. For the convenience of open source, we recommend using the latter.
- To make changes to the raw data, you can use toolkits like [music21](https://github.com/cuthbertLab/music21), [miditoolkit](https://github.com/YatingMusic/miditoolkit), or [MuseScore](https://github.com/musescore/MuseScore).
- To process in the data flow, you can use score [readers and writers](https://github.com/espnet/espnet/tree/master/espnet2/fileio/score_scp.py) provided. Examples can be found in functioin `make_segment` from `egs2/{natsume, ameboshi, pjs}/svs1/local/{prep_segments.py, prep_segments_from_xml.py}/`.

Below are some common errors to watch out for:

#### 1. Wrong segmentation point
* Add pauses or directly split between adjacent lyrics.
* Remove pauses and assign the duration to correct phoneme.

#### 2. Wrong lyric / midi annotation
* Replace with correct one.
* Add missing one and reassign adjacent duration.
* Remove redundant one and reassign adjacent duration.

#### 3. Different lyric-phoneme pairs against the given g2p
* Use a `customed_dic` of syllable-phoneme pairs as following:
    ```
    # e.g.
    # In Japanese dataset ofuton, the output of "ヴぁ" from pyopenjtalk is different from raw data "v a"
    > pyopenjtalk.g2p("ヴぁ")
    v u a
    # Add the following lyric-phoneme pair to customed_dic
    ヴぁ v_a
    ```
* Specify `--g2p none` and store the lyric-phoneme pairs into `score.json`, especially for polyphone problem in Mandarin.
    ```
    # e.g.
    # In Mandarin dataset Opencpop, the pronounce the second "重" should be "chong".
    > pypinyin.pinyin("情意深重爱恨两重", style=Style.NORMAL)
    [['qing'], ['shen'], ['yi'], ['zhong'], ['ai'], ['hen'], ['liang'], ['zhong']]
    ```
#### 4. Special marks in MusicXML
* Breath:
  * `breath mark` in note.articulations: usually appears at the end of the sentence. In some situations, `breath mark` doesn't take effect in its belonging note. Please handle them under local/.
  * `br` in note.lyric. (solved in XMLReader)
  * Special note with a fixed special pitch. (solved in XMLReader)
* Staccato: In some situations, there is a break when `staccato` occurs in note.articulations. We let users to decide whether to perform segmentation under local/.


## Supported text cleaner

You can change via `--cleaner` option in `svs.sh`.

- `none`: No text cleaner.

You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_cleaner.py).

## Supported text frontend

You can change via `--g2p` option in `svs.sh`.

- `none`: Just separate by space
    - e.g.: `HH AH0 L OW1 <space> W ER1 L D` -> `[HH, AH0, L, OW1, <space>, W, ER1, L D]`
- `pyopenjtalk`: [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
    - e.g. `こ、こんにちは` -> `[k, o, pau, k, o, N, n, i, ch, i, w, a]`


You can see the code example from [here](https://github.com/espnet/espnet/blob/cd7d28e987b00b30f8eb8efd7f4796f048dc3be9/test/espnet2/text/test_phoneme_tokenizer.py).

## Supported pretrained model

- [Self supervised learning model: HuBERT, WavLM, wav2vec2.0, ...](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html)
- Codec model

## Supported Models

You can train the following models by changing `*.yaml` config for `--train_config` option in `run.sh`.

- [Naive-RNN](https://arxiv.org/abs/2010.12024)
- [XiaoiceSing](https://arxiv.org/abs/2006.06261)
- [TokSing](https://arxiv.org/abs/2406.08416)

You can find example configs of the above models in [`egs/opencpop/svs2/conf/tuning`](../../opencpop/svs2/conf/tuning).
