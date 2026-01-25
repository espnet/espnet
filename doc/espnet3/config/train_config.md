---
title: ESPnet3 Train Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🧩 ESPnet3 Train Configuration

This page explains the `train.yaml` schema used by the train stage.
For how configs map to stages, see [Config overview](./index.md).

## Minimum required keys (typical train run)

The exact required keys depend on your system and model, but a standard
train stage usually needs:

Required:

- `model` or `task` (one of them) for model construction
- `dataset` (train/valid definitions)
- `dataloader`
- `optim` and `scheduler` (or `optims` and `schedulers`)
- `trainer`
- `exp_dir` and `stats_dir`

Common optional:

- `create_dataset` (only for the create_dataset stage)
- `tokenizer` (ASR-style text processing)
- `num_device`, `num_nodes`
- `dataset_dir`, `data_dir`, `recipe_dir`

Minimal example (custom model path):

```yaml
exp_dir: exp/my_exp
stats_dir: exp/my_exp/stats

model:
  _target_: src.my_model.MyModel

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train: []
  valid: []

dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory: null
    batch_size: 4
    shuffle: true
  valid:
    iter_factory: null
    batch_size: 4
    shuffle: false

optim:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 1000

trainer:
  accelerator: auto
  devices: 1
  max_epochs: 1
```

### ✅ Config sections overview

| Section | Description |
| --- | --- |
| `num_device`, `num_nodes` | Resource counts for training. |
| `task` | Task class and model-related settings used by the ESPnet task stack. |
| `recipe_dir`, `data_dir`, `exp_dir`, ... | Path scaffold for outputs, logs, and cached assets. |
| `create_dataset` | Dataset creation function and parameters for the create_dataset stage. |
| `dataset` | Train/valid dataset splits and DataOrganizer definitions. |
| `tokenizer` | Optional tokenizer text builder/config (mostly for ASR-style recipes). |
| `dataloader` | Collate, iterator, sampler, and sharding settings. |
| `optim`, `scheduler`, `best_model_criterion` | Optimization setup used by training. |
| `trainer`, `fit` | Lightning Trainer arguments and fit-time options. |

### Core config layout

Organise YAML files into small, purpose-driven sections. The example below is
from an ASR recipe, so it includes `tokenizer` settings that may not appear in
all tasks.

```yaml
num_device: 1
num_nodes: 1
task:  # Task class and model-related config

recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: asr_template_train
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
dataset_dir: /path/to/your/dataset

create_dataset:
  func: src.create_dataset.create_dataset
  dataset_dir: ${dataset_dir}

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  train: []
  valid: []

tokenizer:
  text_builder:
    func: src.tokenizer.gather_training_text
    dataset_dir: ${dataset_dir}

dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1

optim:
  _target_: torch.optim.Adam
  lr: 0.002

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 15000

trainer:
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
```

Hydra instantiates each `_target_` at runtime, so the same pattern works for
ESPnet-provided components or your own classes.

### Parallel execution
Parallel execution is documented separately:

- [Provider / Runner](../core/parallel/provider_runner.md)
- [Multi-GPU / multi-node](../core/parallel/multiple_gpu.md)

### Model definition
If `task` is set, ESPnet3 uses the ESPnet2 task-side model definition. This lets
you reuse ESPnet2 recipe configs by placing the familiar model fields under
`model` and keeping the ESPnet2-style structure intact.

If you want a custom model (or an ESPnet3-only model that is not part of the
ESPnet2 task stack), leave `task` unset and instantiate the model directly via
Hydra.


Two common patterns:

1. **Reuse ESPnet models**
   ```python
   from omegaconf import OmegaConf

   from espnet3.components.modeling.lightning_module import ESPnetLightningModule
   from espnet3.utils.task_utils import get_espnet_model

   model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
   espnet_model = get_espnet_model(cfg.task, model_cfg)
   lit_model = ESPnetLightningModule(espnet_model, cfg)
   ```

2. **Instantiate custom models**
   ```python
   from hydra.utils import instantiate

   from espnet3.components.modeling.lightning_module import ESPnetLightningModule

   custom_model = instantiate(cfg.model)
   lit_model = ESPnetLightningModule(custom_model, cfg)
   ```

Both feed directly into the Lightning trainer specified by `trainer`.

### Optimisers and schedulers

ESPnet3 supports single or multiple optimiser setups. See
`../core/components/optimizer_configuration.md` for the rules enforced by
`ESPnetLightningModule.configure_optimizers` (matching parameter groups, scheduler
counts, etc.).

### Dataloader configuration

`dataloader` controls the collate function and the iterator strategy for
`train` and `valid`. ESPnet3 supports two paths:

1. **ESPnet iterator (SequenceIterFactory)**: use `iter_factory` + `batches`
   to build sequence-based batch samplers (the default in templates).
2. **Standard PyTorch DataLoader**: set `iter_factory: null` and use
   `batch_size`, `num_workers`, and `shuffle`.

**SequenceIterFactory example**

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    multiple_iterator: false
    num_shards: 1
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 4
        batch_bins: 4000000
  valid:
    multiple_iterator: false
    num_shards: 1
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/valid/feats_shape
        batch_size: 4
        batch_bins: 4000000
```

**Standard DataLoader example**

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory: null
    batch_size: 4
    num_workers: 2
    shuffle: true
  valid:
    iter_factory: null
    batch_size: 4
    num_workers: 2
    shuffle: false
```

To enable sharded datasets, set `multiple_iterator: true` and `num_shards`,
and use `{shard_idx}` in `shape_files` so the builder can swap in the shard
index at runtime.

### Trainer parameters

Most fields under `trainer` are passed straight to `lightning.pytorch.Trainer`.
Objects that need construction (loggers, callbacks, profilers) should be listed
with `_target_` entries so ESPnet3 can instantiate them before handing them to
Lightning.

```yaml
trainer:
  accelerator: gpu
  devices: 4
  num_nodes: 1
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  log_every_n_steps: 500
  max_epochs: 70

  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger
    - _target_: lightning.pytorch.loggers.WandbLogger
      project: espnet3
      name: ${exp_tag}
      save_dir: ${exp_dir}/wandb

  callbacks:
    # This AverageCheckpointsCallback is included as a default callback without writing here.
    # We included this as an example.
    - _target_: espnet3.components.callbacks.AverageCheckpointsCallback
      output_dir: ${exp_dir}
      best_ckpt_callbacks: []

fit:
  ckpt_path: ${exp_dir}/checkpoints/last.ckpt
```

With this structure, ESPnet3 configurations stay modular while scaling from
single-GPU experiments to large cluster runs with minimal changes.
