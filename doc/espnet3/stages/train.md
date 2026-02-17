---
title: ESPnet3 Train Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Train Stage

The `train` stage runs model training based on `train.yaml` using
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html).

## Quick usage

### Run

```bash
python run.py --stages train --train_config conf/train.yaml
```

### Configure (in `train.yaml`)

Keep the core settings in `train.yaml`:

- `task`, `model`, `dataset`, `dataloader`
- `trainer`, `optim`, `scheduler`
- `exp_dir`, `stats_dir`

For the full configuration list, see the
[train config reference](../config/train_config.md).

| Section | Description |
| --- | --- |
| `task` | Task entrypoint (enables ESPnet2 models). |
| `model` | Model definition and normalization settings. |
| `dataset` | Train/valid splits and dataset classes. |
| `dataloader` | Collate + iterator settings. |
| `trainer` | PyTorch Lightning trainer configuration. |
| `optim` | Optimizer definition. |
| `scheduler` | Learning rate schedule. |
| `exp_dir` | Training output directory. |
| `stats_dir` | Stats output directory (collect_stats). |

### Outputs

Typical outputs are written under:

- `exp_dir`: checkpoints, logs (including TensorBoard), saved configs
- `stats_dir`: feature shapes and global stats (from `collect_stats`)

## Developer Notes

### Details by topic

### [Dataset](./train/dataset.md)

Dataset uses `DataOrganizer` to define train/valid splits.
<!-- TODO(masao): link to GitHub DataOrganizer once PR is merged. -->

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train:
    - name: train_nodev
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_nodev.tsv
  valid:
    - name: train_dev
      dataset:
        _target_: src.dataset.MiniAN4Dataset
        manifest_path: ${dataset_dir}/manifest/train_dev.tsv
```

### [Dataloader + Collate](./train/dataloader.md)

Dataloader defines `collate_fn` and iterator behavior.
You can use the ESPnet iterator setup (expected by collect_stats), or switch to
the standard PyTorch DataLoader if preferred.

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape
```

### [Trainer](../core/components/trainer.md)

Trainer config maps to the Lightning Trainer.

```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
  log_every_n_steps: 100
  gradient_clip_val: 1.0
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger
```

### [Optimizer + Scheduler](./train/optim_scheduler.md)

Optimizer and scheduler control updates and LR schedule.
Multiple optimizers are supported when needed.

```yaml
optim:
  _target_: torch.optim.Adam
scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 15000
```

### [Model](../core/components/model.md)

Model defines the network and optional normalization.
Both ESPnet2-derived models (via `task`) and custom models are supported.

```yaml
model:
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
  encoder: transformer
  decoder: transformer
```

### [Callbacks](../core/components/callbacks.md)

Callbacks let you inject custom behaviors into training.

```yaml
callbacks:
  - _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${exp_dir}/checkpoints
    save_top_k: 3
    monitor: valid/loss
    mode: min
```
