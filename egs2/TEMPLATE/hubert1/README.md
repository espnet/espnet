# Self-supervised Learning

This is a template of the hubert1 recipe for ESPnet2, designed for HuBERT-style SSL.

## Table of Contents
- [Recipe flow](#recipe-flow)
    - [Download and preprocess data](#1-download-and-preprocess-data)
    - [Speed Perturbation](#2-speed-perturbation)
    - [Format wav](#3-format-wav)
    - [Remove long/short data](#4-remove-longshort-data)
    - [Create pseudo-label for the next iteration](#5-create-pseudo-label-for-the-next-iteration)
        - [Feature Dumping](#feature-dumping)
        - [Train K-mean Clustering](#train-k-mean-clustering)
        - [Generate K-mean pseudo-labels](#generate-k-mean-pseudo-labels)
        - [Evaluate Qualities of pseudo-label](#evaluate-qualities-of-pseudo-label)
        - [Prepare a dictionary for training](#prepare-a-dictionary-for-training)
    - [Collect Hubert Statistic](#6-collect-hubert-statistic)
    - [Train Hubert](#7-train-hubert)
    - [Model Packing](#8-model-packing)
    - [Upload Model to HuggingFace](#9-upload-model-to-huggingface)
- [Distillation](#distillation)
    - [DiceHubert](#dicehubert)
- [Evaluation](#evaluation)
- [Differences from other recipes](#differences-from-other-recipes)

## Recipe flow

### 1. Download and preprocess data
This stage handles the data preparation step. It calls `local/data.sh` to download the raw corpus and organize it into Kaldi-style directories within `data/`.

### 2. Speed Perturbation
If `speed_perturb_factors` are defined, the recipe generates time-stretched/compressed versions of the audio. These are stored in a new directory and merged into `data/${train_set}_sp` to increase dataset diversity.

### 3. Format wav
The HuBERT recipe supports only `--feat_type raw`. This stage reformats the audio, such as resampling, segmentation, and format conversion, and saves the processed wav.scp to `dump/raw/`.

### 4. Remove long/short data
This stage removes utterances that are too short or too long based on `--min_wav_duration` and `--max_wav_duration`.

### 5. Create pseudo-label for the next iteration
HuBERT training is iterative. Stages 5 through 7 are repeated for each iteration. This stage calls `scripts/feats/perform_kmeans.sh` and creates the pseudo-label that the model will learn to predict. It is broken down into five sub-steps:

#### Feature Dumping
This step extracts features, which can be either MFCC or SSL features from a previous iteration. You can specify the feature type using `--features_km` and select the layer using `--layers_km`. For example, in the default HuBERT setup:
- First iteration uses MFCC
- Later iterations use HuBERT features from layers 6 and 9
Then, we can set as follows:
```sh
--features_km "mfcc espnet_hubert espnet_hubert"
--layers_km "0 6 9"
```
The extracted features are then saved in `dump/hubert_feats/`.

#### Train K-mean Clustering
This step trains a k-means model on the extracted features. You can set the number of clusters using `n_clusters_lists`, following the same format as `--features_km`. To reduce computation, you can train on a subset of the data using `--portion_km`. The trained model is saved in the `exp/` directory.

#### Generate K-mean pseudo-labels
The trained k-means model assigns a cluster ID to each feature vector. These cluster IDs serve as pseudo-labels. The format of this pseudo-label is `utt_id 12 15 ...` where each number represents the cluster ID of a frame.

#### Evaluate Qualities of pseudo label
If phoneme labels are available, this step evaluates the quality of pseudo-labels against the phoneme labels.

#### Prepare a dictionary for training
This step sorts cluster IDs by frequency and saves the resulting dictionary in the `data/` directory.

### 6. Collect Hubert Statistic
This stage computes statistics for training. It collects input/output shapes and calculates normalization statistics (mean and variance) over the training and validation sets.

### 7. Train Hubert
This stage trains the HuBERT model. You can modify training settings using `--train_configs`. Multiple configurations can be provided (one per iteration) using space-separated values, similar to `--features_km`. For default settings, refer to the HuBERT recipe for the LibriSpeech dataset.

### 8. Model Packing
This stage packs the trained model files for later use or distribution.

### 9. Upload Model to HuggingFace
This stage uploads the trained model to Hugging Face.

## Distillation

### DiceHubert
[DiceHuBERT](https://arxiv.org/pdf/2507.02911) is a distillation method that transfers knowledge from a teacher model to a smaller student model. Unlike conventional approaches that use regression losses, DiceHuBERT applies the standard HuBERT cross-entropy loss for distillation. This design makes it compatible with the HuBERT recipe with only minor changes.

We start with the standard data preparation:
```sh
./run.sh --stage 1 --stop-stage 4
```
Following the original HuBERT setup, the student model is trained using the same procedure as iteration 2, with a smaller model size. To generate pseudo-labels without training HuBERT from scratch, you can download a pretrained model as follows:
```sh
./run.sh --stage 5 --stop-stage 5 --train_start_iter 2 --train_stop_iter 2 --download_model simpleoier/simpleoier_librispeech_hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw
```
Setting --train_start_iter 2 --train_stop_iter 2 ensures that only iteration 2 is executed. After generating pseudo-labels, continue with the standard training steps:
```sh
./run.sh --stage 6 --stop-stage 7 --train_start_iter 2 --train_stop_iter 2
```
The default DiceHuBERT configuration is available at `librispeech/hubert1/conf/tuning/train_ssl_torchaudiohubert_distill_960h_pretrain_it2.yaml`.

## Evaluation
This recipe does not include a built-in evaluation stage. However, the trained HuBERT model is compatible with the SUPERB benchmark.

First, clone the [s3prl repository](https://github.com/s3prl/s3prl/tree/main). Since `s3prl` supports HuBERT models, you can set the upstream model to `espnet_hubert_local` as follows:
```sh
python s3prl/run_downstream.py \
    -m train \
    -d $TASK \
    -u espnet_hubert_local \
    -k "$CKPT" \
    -g "$CONFIG"
```
where `$CKPT` and `$CONFIG` refer to the HuBERT checkpoint and configuration files located in the `exp/` directory. Running this script trains a downstream model using features extracted from your HuBERT model for a specific task. You can then evaluate the downstream model using the standard `s3prl` pipeline. For more details, please refer to the [SUPERB document](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md#asv-automatic-speaker-verification)


## Differences from other recipes

ESPnet2 serves two different recipes for Self-Supervised Learning (SSL): `ssl1` and `hubert1` (this one).

`hubert1` is the original implementation of SSL under the [HuBERT](https://arxiv.org/abs/2106.07447) pre-training framework. The recipe takes care of everything need for pre-training, such as K-means pseudo-labelling and discrete token evaluation. This is very important for reproducibility. However, it is quite complicated due to the multiple offline stages required for HuBERT and therefore difficult to hack/adapt to new training methods or other scenarios.

We created the new `ssl1` recipe to future-proof the codebase to accomodate other pre-training techniques that are purely end-to-end, such as [DinoSR](https://arxiv.org/abs/2305.10005), [SpeechFlow](https://arxiv.org/abs/2310.16338), or [w2v-BERT](https://arxiv.org/abs/2108.06209). This recipe is designed to be easily customizable and more scalable to large-scale pre-training setups.

Note: the `ssl1` codebase also supports HuBERT pre-training, but the steps to create the pseudo-labels are not included in the recipe. Users will either need to run the `hubert1` recipe to obtain the labels, or generate it themselves.
