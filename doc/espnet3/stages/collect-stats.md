---
title: ESPnet3 Collect Stats Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Collect Stats Stage

The `collect_stats` stage computes dataset statistics (feature shapes and global
stats) used by training and normalization. For background, motivation, and
advanced use cases, see
[Collect Stats Phase Overview](./collect_stats_description.md).

## Quick usage

### Run

```bash
python run.py --stages collect_stats --train_config conf/train.yaml
```

This runs `collect_stats` over the **train** and **valid** splits. Outputs are
written under `stats_dir/train` and `stats_dir/valid`.

### Configure (in `train.yaml`)

`collect_stats` reads the `train.yaml` used for training. At minimum:

- `stats_dir` must be set (outputs are written here).
- `dataset` and `dataloader` define which splits and batching to process.
- `model.normalize_conf.stats_file` can point to the produced stats file.

Example config (lightweight):

```yaml
stats_dir: ${exp_dir}/stats

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train:
    - name: train
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_nodev.tsv
  valid:
    - name: valid
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_dev.tsv

dataloader:
  train:
    iter_factory:
      batches:
        shape_files:
          - ${stats_dir}/train/feats_shape

model:
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

Notes:

- `collect_stats` only processes `train` and `valid`; `test` is ignored.
- During `collect_stats`, the value of `model.normalize_conf.stats_file` is
  ignored; stats are always written under `stats_dir`.
- If `model.normalize_conf.stats_file` points into `stats_dir` and the file
  already exists, it will be overwritten by this stage.

### Outputs

`collect_stats` writes files under `stats_dir` per split:

```
${stats_dir}/
  train/
    feats_shape
    feats_stats.npz
    stats_keys
  valid/
    feats_shape
    feats_stats.npz
    stats_keys
```

## Developer Notes

### What runs under the hood

`collect_stats` instantiates the model and trainer, then calls `trainer.collect_stats()`:

```python
def collect_stats(cfg):
    _ensure_directories(cfg)
    trainer = _build_trainer(cfg)
    trainer.collect_stats()
```

The model's `collect_stats()` uses the dataset and dataloader configs to gather
feature shapes and aggregate sums/squares via
`espnet3.components.data.collect_stats.collect_stats`.
<!-- TODO(masao): link to GitHub source once PR is merged. -->

`trainer.collect_stats()` ultimately calls the model's `collect_feats()` to
extract features used for statistics. If you set `task` in `train.yaml` and use
ESPnet2-derived models, `collect_feats()` is already implemented, so no extra
work is needed.

If you implement a custom model, add a `collect_feats()` method with the same
contract:

- **Inputs**: keyword arguments matching the batch dictionary from your
  `collate_fn` (e.g., `speech`, `speech_lengths`, `text`, `text_lengths`).
- **Output**: a dict of tensors keyed by feature name, with optional
  `*_lengths` entries. For example, ASR models return:
  `{"feats": feats, "feats_lengths": feats_lengths}`.

This is an ASR-style example; for ASR datasets, `speech` and `text` are expected
to be provided by the dataset class (see
[Dataloader + Collate](./train/dataloader.md)).

Sample custom model:

```python
class MyCustomModel:
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        pass

    def collect_feats(self, speech, speech_lengths, **kwargs):
        feats = speech
        # *_lengths are populated by the ESPnet collate function.
        feats_lengths = speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}
```
