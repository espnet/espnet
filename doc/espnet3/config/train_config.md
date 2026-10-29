---
title: ESPnet3 Training Configuration
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-15
---

# ESPnet3 Training Configuration

This page describes the `training` config role used by the ASR system.
The [configuration/launch guide](./index.html) explains default loading,
whole-key inheritance, and the distinction between config roles and stages.
It configures the following stages:
- `train_tokenizer`
- [`create_dataset`](../stages/create-dataset.html)
- [`collect_stats`](../stages/collect-stats.html)
- [`train`](../stages/train.html)


## Overview

| Section                                            | Required       | Description                                                     |
| -------------------------------------------------- | -------------- | --------------------------------------------------------------- |
| `recipe_dir`, `data_dir`, `exp_dir`, ...           | ✅              | path scaffold for outputs and cached assets                     |
| `num_device`, `num_nodes`                          |                | resource counts for training                                    |
| `task`                                             |                | ESPnet task entrypoint; optional — selects task-loader model resolution instead of direct Hydra instantiation of `model` |
| `model`                                            | ✅              | Model definition; always required — interpreted as an ESPnet2-style task model when `task` is set, otherwise instantiated directly via Hydra |
| `create_dataset`                                   |                | dataset builder kwargs used by `create_dataset`                 |
| `dataset`                                          | ✅              | train and valid dataset definitions resolved by `DataOrganizer` |
| `tokenizer`                                        |                | tokenizer or text-builder settings                              |
| `dataloader`                                       | ✅              | collate, iterator, sampler, and sharding settings               |
| `optimizer`/`optimizers`, `scheduler`/`schedulers` | ✅              | optimization setup                                              |
| `trainer`                                          | ✅              | Lightning trainer arguments                                     |
| `fit`                                              |                | Lightning fit-time options                                      |
| `parallel`                                         |                | parallel processing settings                                    |
| `best_model_criterion`                             |                | checkpoint-selection criteria for callbacks                     |


## Path Scaffold

This section defines the paths used during training.

### Default values

| Key          | Description                                              | Default value                  |
| ------------ | -------------------------------------------------------- | ------------------------------ |
| `num_device` | Number of devices used in training                       | `1`                            |
| `num_nodes`  | Number of nodes used in training                         | `1`                            |
| `recipe_dir` | Path to the recipe directory                             | `.`                            |
| `data_dir`   | Path to the raw data directory                           | `${recipe_dir}/data`           |
| `exp_tag`    | Identifier used to name the experiment                   | `${self_name:}`                |
| `exp_dir`    | Path to the experiment directory                         | `${recipe_dir}/exp/${exp_tag}` |
| `stats_dir`  | Path to where the outputs of `collect_stats` are written | `${recipe_dir}/exp/stats`      |

`exp_tag` participates directly in experiment directory naming. The baseline
TEMPLATE `stats_dir` is shared across tags under `exp/stats`; it is not
`${exp_dir}/stats`. Override it explicitly for experiments that must keep
incompatible statistics separate, and point `model.normalize_conf.stats_file`
and sampler shape paths to the same chosen directory.

By default, TEMPLATE `training.yaml` uses:

```yaml
exp_tag: ${self_name:}
```

That means `exp_tag` defaults to the config filename. For example,
`training_e_branchformer.yaml` resolves to:

```yaml
exp_tag: training_e_branchformer
```


See [Resolvers](#resolvers) below for `self_name` and other custom resolvers.

### Example
```yaml
num_device: 1
num_nodes: 1

recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
dataset_dir: /path/to/your/dataset
```

## Core config layout

This section should be read as a user-authored override config, not as the full
TEMPLATE default.

The reviewed launcher keeps only explicitly requested stage entries.
ASR `train()` reads the `train_tokenizer` entry even when a tokenizer is
already cached, so include it in the stage list:

```bash
python run.py --stages train_tokenizer train \
  --training_config conf/training_asr_transformer.yaml
```

Run dataset creation and statistics first, or include them before training
as shown in the full pipeline in the [launch guide](./index.html).

Most recipes keep the default path scaffold from `egs3/TEMPLATE/esp2_asr/conf/training.yaml` and only override the task-specific parts they need.

Example:

```yaml
task: espnet3.systems.esp2_asr.task.ASRTask

create_dataset:
  recipe_dir: ${recipe_dir}

dataset:
  train:
    - data_src_args:
        split: train
  valid:
    - data_src_args:
        split: valid

tokenizer:
  vocab_size: 5000

dataloader:
  train:
    iter_factory:
      batches:
        type: sorted
        batch_size: 16

optimizer:
  lr: 0.002

scheduler:
  warmup_steps: 15000

trainer:
  log_every_n_steps: 100
  max_epochs: 10
```


## `model`

If `task` is set, ESPnet3 uses the ESPnet2 task-side model definition. This is
the normal way to reuse ESPnet2-style model config blocks.

If you want a custom model, leave `task` unset and instantiate the model
directly via Hydra in `model`.

Example with `task`:

```yaml
task: espnet3.systems.esp2_asr.task.ASRTask

model:
  frontend: default
  encoder: e_branchformer
  decoder: transformer
  normalize: global_mvn
  normalize_conf:
    stats_file: ${stats_dir}/train/feats_stats.npz
```

In this case, `model` is interpreted as the task-side model config.
This is usually the copy-and-adapt path from an ESPnet2 recipe config.

Example without `task`:

```yaml
task:

model:
  _target_: my_project.models.MyASRModel
  vocab_size: 5000
  hidden_size: 256
```

In this case, `model._target_` is required because ESPnet3 instantiates the
model directly through Hydra.

## `create_dataset`

`create_dataset` is the config block for the `create_dataset` stage. For each
unique dataset source referenced under `dataset.train` / `dataset.valid` /
`dataset.test`, `BaseSystem.create_dataset()` loads the recipe's
`dataset/__init__.py:DatasetBuilder` class and calls, in order,
`is_source_prepared`, `prepare_source`, `is_built`, and `build` — all keyword
arguments in this block are forwarded to every one of those calls.

See these pages for details:

- [Create dataset stage](../stages/create-dataset.html)
- [Dataset references and builders](https://github.com/espnet/espnet/blob/master/espnet3/components/data/data_organizer.py)

### Settings

| Key           | Description                                          |
| ------------- | ----------------------------------------------------- |
| `recipe_dir`  | Recipe directory, forwarded as a builder kwarg         |
| `dataset_dir` | Dataset directory, forwarded as a builder kwarg        |

Note: the TEMPLATE header comment also shows a `create_dataset.func` key. It is
not read by the current stage implementation — the stage always resolves the
builder class from `dataset/__init__.py`, so leave `func` unset.

### Example

```yaml
create_dataset:
  recipe_dir: ${recipe_dir}
  dataset_dir: ${dataset_dir}
```

## `dataset`

Dataset entries use `DataOrganizer` and dataset references.

Each dataset entry may resolve by:

- dataset tag
- explicit module path
- omitted `data_src` -> `${recipe_dir}/dataset/__init__.py`
- local recipes often omit `data_src` and use `${recipe_dir}/dataset/__init__.py`

Only `data_src_args` is passed to `Dataset(...)`. `name` and `transform`
belong to the organizer. Recipe dataset classes inherit `BaseDataset` and
declare their raw item `fields`; see [contracts](./contract.html#dataset-declaration-belongs-on-the-class).
The inference role supplies its own `dataset` top-level block, replacing
rather than deep-extending the training dataset block.

See [Dataset references and builders](https://github.com/espnet/espnet/blob/master/espnet3/components/data/data_organizer.py) for
`data_src` details.

### Example
```yaml
dataset:
  recipe_dir: ${recipe_dir}
  train:
    - data_src_args:
        split: train
    - data_src: egs3.librispeech_100.esp2_asr.dataset
      data_src_args:
        split: train-clean-100
  valid:
    - data_src_args:
        split: valid
```

## `dataloader`

Two common modes:

1. ESPnet iterator mode through `iter_factory`
2. plain PyTorch DataLoader mode with `iter_factory: null`

See [Dataloader and Collate](https://github.com/espnet/espnet/blob/master/espnet3/components/data/dataloader.py) for `iter_factory`
details, supported iterator factories, and full config examples.

### Examples
Sequence Iterator:

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
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 4
        batch_bins: 4000000
```

`multiple_iterator` in ESPnet2 is not supported in current ESPnet3.

Standard DataLoader:

The following shows the desired **effective mapping**, not a complete override
of the shipped TEMPLATE. The TEMPLATE also contains dataloader-level
`total_shards`/`dist_world_size` keys; remove those inherited keys before
passing the mapping to `DataLoaderBuilder`. Setting them to `null` does not
remove them, and PyTorch DataLoader rejects them.

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

### Sharding keys

Do not set `total_shards` / `dist_world_size` under `dataloader.train` /
`dataloader.valid` — `DataLoaderBuilder._maybe_shard_dataset()` reads them as
attributes of the underlying dataset object instead, so a dataloader-level
copy is silently ignored in `iter_factory` mode and raises a `TypeError` in
standard-DataLoader mode (`iter_factory: null`), where leftover
`dataloader.train`/`dataloader.valid` keys are forwarded straight to
`torch.utils.data.DataLoader(...)`. Set them on the dataset instead (e.g. via
`data_src_args`) as described in
[Dataset Sharding](https://github.com/espnet/espnet/blob/master/espnet3/components/data/dataset.py); for a
`ShardedDataset`, set `total_shards` / `dist_world_size` on that dataset.
An ordinary, unsharded dataset does not require these attributes.

## `optimizer` / `scheduler`


`scheduler_interval` and `scheduler_monitor` work as follows:

| Tag                         | Description                                                          |
| --------------------------- | -------------------------------------------------------------------- |
| `scheduler_interval: step`  | step the scheduler after optimizer updates                           |
| `scheduler_interval: epoch` | step the scheduler at epoch boundaries                               |
| `scheduler_monitor`         | metric name used by monitored schedulers such as `ReduceLROnPlateau` |

Notes:

- `step` is the common choice for schedulers such as `WarmupLR`
- `epoch` is used when the scheduler should react once per epoch
- `scheduler_monitor` is only needed for schedulers that require a monitored value
- use the same metric key that appears in logs, for example `valid/loss`

Named multi-optimizer path:

See [Multiple optimizers and schedulers](https://github.com/espnet/espnet/blob/master/espnet3/components/modeling/optimization_spec.py)
for the full behavior.

### Default Values

| Key                      | Default value                           |
| ------------------------ | --------------------------------------- |
| `optimizer._target_`     | `torch.optim.Adam`                      |
| `optimizer.lr`           | `0.002`                                 |
| `optimizer.weight_decay` | `0.000001`                              |
| `scheduler._target_`     | `espnet2.schedulers.warmup_lr.WarmupLR` |
| `scheduler.warmup_steps` | `15000`                                 |
| `scheduler_interval`     | `step`                                  |
| `parallel.env`           | `local`                                 |
| `parallel.n_workers`     | `1`                                     |

### Examples

Single Optimizer:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 1000

scheduler_interval: step
scheduler_monitor:
```

Multiple Optimizers:

The model must return named `OptimizationStep` objects for the configured
optimizers. Checkpoint and monitored-scheduler keys must match the names logged
by that model; a default `valid/loss` monitor is not automatically a combined
multi-optimizer loss. An example with named generator/discriminator losses:

```yaml
best_model_criterion:
  - - valid/generator/loss
    - 5
    - min

optimizers:
  generator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: generator
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm

  discriminator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: discriminator

schedulers:
  generator:
    scheduler:
      _target_: torch.optim.lr_scheduler.LinearLR
      total_iters: 1000
    interval: step

  discriminator:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
    interval: epoch
    monitor: valid/discriminator/loss
```

## `trainer`

`trainer` maps to Lightning trainer construction through
`ESPnet3LightningTrainer` (the `trainer` subtree, not the whole training config).

Example:

```yaml
trainer:
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
  max_epochs: 10
  log_every_n_steps: 100
```

In multi-optimizer mode, trainer-level gradient clipping should not be used.
See [Multiple optimizers and schedulers](https://github.com/espnet/espnet/blob/master/espnet3/components/modeling/optimization_spec.py)
for details.

## `parallel`

This section configures parallel execution. Details are documented here:

- [Provider / Runner](../parallel.html)
- [Multi-GPU / multi-node](../parallel.html#lightning-training)

### Default Values

| Key                  | Default value |
| -------------------- | ------------- |
| `parallel.env`       | `local`       |
| `parallel.n_workers` | `1`           |

### Examples

Minimal local example:

```yaml
parallel:
  env: local
  n_workers: 1
```

Minimal SLURM example:

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```


## `fit`

`training_config.fit` is forwarded to `trainer.fit(...)`.

This is where runtime fit-time overrides belong.


### Example

Resume from checkpoint:

```yaml
fit:
  ckpt_path: ${exp_dir}/last.ckpt
```

## Other Training Settings

### Settings

| Key                    | Description                                |
| ---------------------- | ------------------------------------------ |
| `init`                 | Weight initialization strategy             |
| `seed`                 | Optional random seed for `collect_stats`/`train` |
| `best_model_criterion` | Criteria used to compare model performance |

The TEMPLATE writes `init` at the top level, but the reviewed
`ESPnet3LightningTrainer` receives only `training_config.trainer` and reads
`self.config.init`. Therefore the top-level value is **not applied** by this
baseline. Do not assume the example below changes initialization without
checking the trainer path; later fixes to forwarding are outside this
implementation reference.

### Example
```yaml
init: xavier_uniform

best_model_criterion:
  - - valid/loss
    - 10
    - min
```

## Resolvers

ESPnet3 registers a few custom OmegaConf resolvers (see
[`config_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/config_utils.py)).
They are rewritten to plain values while a config is loaded, before any
`${...}` interpolation is resolved, so they only work through
`load_and_merge_config`/`load_config_with_defaults` (i.e. through `run.py`),
not through a bare `OmegaConf.load(...)`.

| Resolver | Usage | Resolves to |
| --- | --- | --- |
| `${self_name:}` | `exp_tag: ${self_name:}` | stem of the config file being loaded, e.g. `training` for `training.yaml`, or `training_e_branchformer` for `training_e_branchformer.yaml` |
| `${config_path:relpath}` | `readme: ${config_path:../src/hf_model_readme.md}` | absolute path, resolved relative to the directory containing the config that references it |
| `${set_corpus_and_system:}` | `hf_repo: espnet/${set_corpus_and_system:}_${exp_tag}` | `<corpus>_<system>` derived from the `egs3/<corpus>/<system>/conf/...` path of the loaded config, e.g. `mini_an4_esp2_asr` |
| `${load_line:relpath}` | `vocab: ${load_line:conf/tokens.txt}` | list of stripped lines read from the given text file |

## Related pages

- [Train stage](../stages/train.html)
- [Create dataset stage](../stages/create-dataset.html)
- [Dataset references and builders](https://github.com/espnet/espnet/blob/master/espnet3/components/data/data_organizer.py)
- [Optimizer configuration](https://github.com/espnet/espnet/blob/master/espnet3/components/modeling/lightning_module.py)
- [Dataset Sharding](https://github.com/espnet/espnet/blob/master/espnet3/components/data/dataset.py)
