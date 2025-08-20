# Data Preparation Scripts Guide

This directory contains scripts for preparing multiple language identification datasets. Please follow the step-by-step instructions below.

## Important Notes

- It's recommended to prepare datasets one by one for easier troubleshooting

## Dataset Preparation Workflow

### 1. Dataset Download

**Manual download required:**
- **BABEL**: Download from [LDC](https://catalog.ldc.upenn.edu)
- **VoxPopuli**: Follow instructions at https://github.com/facebookresearch/voxpopuli

**Automatic download (handled by scripts):**
- **FLEURS, ML-SUPERB2**: Downloaded from Hugging Face
- **VoxLingua107**: Auto-download script provided

### 2. Individual Dataset Preparation

Before running, edit the dataset path configuration in each script:

```bash
# Prepare BABEL dataset
bash local/prepare_babel.sh --dataset_path "/path/to/babel"

# Prepare FLEURS dataset
bash local/prepare_fleurs.sh

# Prepare ML-SUPERB2 dataset
bash local/prepare_ml_superb2.sh

# Prepare VoxLingua107 dataset
bash local/prepare_voxlingua107.sh

# Prepare VoxPopuli dataset
bash local/prepare_voxpopuli.sh
```

### 3. Data Preprocessing

Run the data preprocessing stages (stage 3-4) for each dataset:

```bash
# Example: Process FLEURS dataset
./run_combined.sh --stage 3 --stop_stage 3 \
    --train_set train_fleurs_lang \
    --valid_set dev_fleurs_lang \
    --test_sets test_fleurs_lang
```

This will generate:
- `dump/raw/train_fleurs_lang`
- `dump/raw/dev_fleurs_lang`
- `dump/raw/test_fleurs_lang`

Repeat this process for all datasets. The complete list of dataset splits to process:

- **BABEL**: `train_babel_lang`, `dev_babel_lang`
- **FLEURS**: `train_fleurs_lang`, `dev_fleurs_lang`, `test_fleurs_lang`
- **ML-SUPERB2**: `train_ml_superb2_lang`, `dev_ml_superb2_lang`, `dev_dialect_ml_superb2_lang`
- **VoxLingua107**: `train_voxlingua107_lang`, `dev_voxlingua107_lang`
- **VoxPopuli**: `train_voxpopuli_lang`, `dev_voxpopuli_lang`, `test_voxpopuli_lang`

### 4. Dataset Combination

After ensuring all datasets have completed the dump stage processing, run the combination script:

```bash
bash local/combine.sh
```

This will generate the final combined dataset: `dump/raw/train_all_no_filter_lang`

## Workflow Summary

```
Download datasets → Configure paths → Run prepare_*.sh → Run dump stages → Combine datasets
```
