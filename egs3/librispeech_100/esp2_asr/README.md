# LibriSpeech 100h ASR recipe

## Data Preparation

`create_dataset` downloads the five OpenSLR archives (`train-clean-100`,
`dev-clean`, `dev-other`, `test-clean`, `test-other`) into `download/` and
extracts them to `download/LibriSpeech`:

```bash
python run.py --stages create_dataset \
    --training_config conf/tuning/training_e_branchformer.yaml
```

To reuse a corpus you already have, point the recipe at it before running
`create_dataset`; either the `LibriSpeech` directory or its parent works.
Locations are checked in this order and the first that holds a corpus wins:

1. `builder.dataset_path` in `dataset/config.yaml` (default `download`,
   relative to this directory)
2. `source_dir` passed to the builder or dataset
3. the `LIBRISPEECH` environment variable

When all splits are present nothing is downloaded. Missing splits are
downloaded only into `builder.dataset_path`, and an existing archive there is
extracted instead of downloaded again. A corpus found through `source_dir` or
`LIBRISPEECH` is never modified; if it is missing splits, `create_dataset`
reports them.

## Quick start

```bash
# 1) Train the default E-Branchformer model
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Infer
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

## Pretrained Models

- [`conf/tuning/training_e_branchformer.yaml`](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
