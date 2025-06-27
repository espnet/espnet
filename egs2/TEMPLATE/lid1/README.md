# Language Identification

This is a template of the `lid1` recipe for ESPnet2.
It follows a classification-based training/inference pipeline for spoken language identification.
The model is trained as a closed-set classifier over a predefined set of language labels.
Optionally, language embeddings can be extracted and used for downstream analysis, e.g., t-SNE visualization.

## Table of Contents

- [Language Identification](#language-identification)
  - [Table of Contents](#table-of-contents)
  - [Recipe flow](#recipe-flow)
    - [1. Data preparation](#1-data-preparation)
    - [2. Speed perturbation (Optional)](#2-speed-perturbation-optional)
    - [3. Wav format](#3-wav-format)
    - [4. Statistics collection](#4-statistics-collection)
    - [5. LID training](#5-lid-training)
    - [6. Inference and embedding extraction](#6-inference-and-embedding-extraction)
    - [7. Score calculation](#7-score-calculation)
    - [8. t-SNE visualization](#8-t-sne-visualization)
    - [9-10. (Optional) Pack and upload results](#9-10-optional-pack-and-upload-results)
  - [How to run](#how-to-run)
    - [Example: VoxLingua107 training](#example-voxlingua107-training)

## Recipe flow

`lid1` recipe consists of 10 stages.

### 1. Data preparation

Prepares Kaldi-style data directories using `local/data.sh`.

Expected files include:
- `wav.scp`: path to raw audio
- `utt2lang`: utterance-to-language mapping
- `lang2utt`: language-to-utterance mapping (for sampling)
- `segments` (optional): used to extract segments from long recordings

### 2. Speed perturbation (Optional)

Applies offline speed perturbation to the training set using multiple speed factors, e.g., `0.9 1.0 1.1`.

### 3. Wav format

Formats the audio to a consistent format (`wav`, `flac`, or Kaldi-ark) and copies necessary metadata to the working directory.
Required for both training and evaluation sets.

### 4. Statistics collection

Collects input feature shape statistics and language information needed for batching and model configuration.

### 5. LID training

Trains the language identification model using the configuration provided via `--lid_config` and optional arguments in `--lid_args`.
The model is trained to predict the correct language ID for each utterance.

### 6. Inference and embedding extraction

Performs inference on evaluation sets. This stage supports both:
- LID prediction (predicted `utt2lang`)
- Language embedding extraction (utterance-level or averaged per language)
- Optionally saves intermediate outputs

### 7. Score calculation

Computes standard classification metrics (Accuracy, Macro Accuracy) by comparing model predictions with reference `utt2lang`.

### 8. t-SNE visualization

Visualizes the per-language embeddings using t-SNE.

### 9-10. (Optional) Pack and upload results

Packs the trained model and metadata into a zip file and optionally uploads it to Hugging Face Hub for sharing and reproducibility.

## How to run

### Example: VoxLingua107 training

Move to the recipe directory:
```sh
cd egs2/voxlingua107/lid1
```

Edit the following files:
```sh
vim db.sh        # set path to VoxLingua107 dataset
vim cmd.sh       # job scheduling command if using a cluster
vim conf/mms_ecapa_bs3min_baseline.yaml  # model and training configuration (default training configuration)
```

Then run the full pipeline:
```sh
./run.sh
```

This will go through all the stages from data preparation to scoring.
