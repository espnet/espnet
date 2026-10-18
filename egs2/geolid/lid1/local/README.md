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

Install `local/requirements.txt` first. It pins `datasets<4` because the shared
FLEURS preparer uses the `google/xtreme_s` dataset loading script. Set
`VOXLINGUA107` in `db.sh`; pass other local corpus paths as command-line options:

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

Run the waveform preparation stage (stage 3) for each dataset:

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

Babel filtering expects materialized 16 kHz waveforms from stage 3. After dumping
both Babel splits, run `bash local/filter_babel_over_10s.sh`; it keeps utterances
of at least 10 seconds and regenerates the matching language lists.

### 4. Dataset Combination

After ensuring all datasets have completed the dump stage processing, run the combination script:

```bash
bash local/combine.sh
```

The combination script checks for duplicate utterance IDs and matching waveform,
language and duration keys, and leaves the source dumps unchanged. It generates
`lang2utt`, `category2utt`, `utt2dataset` and `dataset2utt`.

This will generate the final combined dataset: `dump/raw/train_all_no_filter_lang`

## Workflow Summary

```
Download datasets → Configure paths → Run prepare_*.sh → Run dump stages → Combine datasets
```

## Shared preparation code

- VoxLingua107 calls `egs2/voxlingua107/lid1/local/data.sh` and links its Python
  preparer. Its standard directories are copied to GeoLID's `_lang` names only
  after preparation completes, avoiding the previous path mismatch.
- FLEURS calls `egs2/fleurs/asr1/local/create_dataset.py`, including its train,
  validation/dev and test splits. The local Python adapter converts its TSVs
  to language labels and waveform manifests.
- ML-SUPERB2 calls `egs2/ml_superb2/asr1/local/download.py`, reusing its language
  normalization and dialect correction; the local adapter reads the resulting
  text labels and waveform paths.
- Babel calls `egs2/babel/asr1/local/prepare_acoustic_training_data.pl` for
  transcription segmentation and audio commands. The local adapter adds language
  prefixes and `utt2lang`; merge outputs are truncated on reruns.

Only the combined training dump needs `dataset2utt` and `utt2dataset`. Validation
uses language balancing (`valid_batch_type: catpow`).

The standard stage-1 entrypoint prepares VoxLingua107 by default. To select
additional corpora (after arranging the locally licensed/downloaded audio):

```bash
./run.sh --stage 1 --stop_stage 1 \
    --local_data_opts "--datasets voxlingua107,fleurs,ml_superb2,babel,voxpopuli"
```

Alternatively, invoke the individual preparation scripts above.
