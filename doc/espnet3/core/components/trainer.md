---
title: ESPnet3 Trainer Configuration (Training)
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Trainer Configuration (Training)

ESPnet3 uses PyTorch Lightning for the `train` stage.
Most fields under `trainer:` in `training.yaml` map directly to Lightning
Trainer arguments.

Upstream reference:

<DocCards :cols="1">
  <DocCard
    title="PyTorch Lightning Trainer"
    desc="Official Lightning trainer reference for the upstream argument surface."
    icon="tabler:external-link"
    href="https://lightning.ai/docs/pytorch/stable/common/trainer.html"
  />
</DocCards>

ESPnet3 wraps Lightning in:

- `espnet3.components.trainers.trainer.ESPnet3LightningTrainer`

## Common settings

| Key | Description |
| --- | --- |
| `accelerator` | backend selection such as `auto`, `gpu`, `cpu` |
| `devices` | number of devices to use |
| `num_nodes` | number of nodes for distributed training |
| `max_epochs` | total training epochs |
| `accumulate_grad_batches` | gradient accumulation steps |
| `gradient_clip_val` | gradient clipping value |
| `gradient_clip_algorithm` | clipping mode such as `norm` |
| `log_every_n_steps` | logging frequency in steps |
| `check_val_every_n_epoch` | validation interval in epochs |
| `precision` | mixed precision setting such as `32`, `bf16-mixed` |
| `strategy` | distributed strategy such as `ddp`, `auto` |
| `logger` | logger configuration such as TensorBoard or CSV |
| `callbacks` | extra training callbacks |
| `profiler` | Lightning profiler config |
| `plugins` | Lightning plugin config |
| `callbacks` | extra callbacks, appended after the ESPnet3 defaults |
| `init` | initialization mode forwarded to `espnet2.torch_utils.initialize.initialize(model, init)` (e.g. `xavier_uniform`); consumed by `ESPnet3LightningTrainer` itself, not forwarded to `lightning.Trainer` |

`accelerator`, `strategy`, `logger`, `profiler`, `plugins`, `callbacks`, and
`init` are consumed by `ESPnet3LightningTrainer` itself and stripped out before
building `lightning.Trainer`. Every other key under `trainer:` is passed
through to `lightning.Trainer(**trainer_config)` unchanged.

**`init` must live under `trainer:`.** `ESPnet3LightningTrainer` only reads
`init` from the trainer sub-config it receives
(`ESPnet3LightningTrainer(config=training_config.trainer, ...)`), i.e. it
looks for `training_config.trainer.init`. A top-level `init:` key at the root
of `training.yaml` (outside `trainer:`) is not read by the current
implementation and has no effect.

## Example

```yaml
trainer:
  accelerator: auto
  devices: ${num_device}
  num_nodes: ${num_nodes}
  max_epochs: 10
  log_every_n_steps: 100
  gradient_clip_val: 1.0
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger
```

## What ESPnet3 adds

Compared with constructing `lightning.Trainer` directly, ESPnet3 adds:

- default callbacks from `get_default_callbacks(...)`
- model initialization through `trainer.init` (see the table above)
- ESPnet dataloader safeguards
- multi-optimizer trainer validation

**Accepted `config` types.** `ESPnet3LightningTrainer.__init__` accepts
`config: Union[DictConfig, Namespace, Dict[str, Any]]`. In practice it reads
and writes attributes on `config` (for example `self.config.log_every_n_steps`,
and it may set `self.config.reload_dataloaders_every_n_epochs` / `
self.config.use_distributed_sampler`), so a plain `dict` currently raises
`AttributeError`. Pass an `OmegaConf`/Hydra `DictConfig` (the normal case when
`training.yaml` is loaded through Hydra) or a `Namespace`.

## Default callbacks

The default stack includes:

- last-checkpoint `ModelCheckpoint`
- best-k `ModelCheckpoint`
- `AverageCheckpointsCallback`
- `LearningRateMonitor`
- `MetricsLogger`
- `TQDMProgressBar`

ESPnet3 adds these automatically.

For details, see [Callbacks](./callbacks.html).

## ESPnet dataloader behavior

When the training dataloader uses ESPnet iterator mode, the wrapper adjusts
Lightning settings to match that path.

Current behavior:

- `reload_dataloaders_every_n_epochs = 1`
- `use_distributed_sampler = False` when ESPnet's sampler is active, or when
  the training dataset shards (`total_shards > 1`) on the standard `DataLoader` path

This keeps Lightning from conflicting with ESPnet's own iterator and sharding
logic.

## Multi-optimizer restrictions

When `config.optimizers` is used, ESPnet3 enables the manual named
multi-optimizer path.

Current trainer-side restrictions:

- trainer-level `gradient_clip_val` is not supported
- trainer-level `gradient_clip_algorithm` is not supported except the default
- DeepSpeed is not supported in the multiple-optimizer path

Use per-optimizer settings under:

- `optimizers.<name>.gradient_clip_val`
- `optimizers.<name>.gradient_clip_algorithm`

See [Multiple optimizers and schedulers](./multiple_optimizers_schedulers.html).



## Main methods

`ESPnet3LightningTrainer` provides:

- `fit(...)`
- `validate(...)`
- `collect_stats(...)`

`fit(...)` and `validate(...)` forward to the underlying Lightning trainer.
`collect_stats(...)` forwards to the model-side stats path.

## Custom trainer example

No recipe currently ships a subclass of `ESPnet3LightningTrainer`; the
standard `espnet3.systems.base.training._build_trainer` always constructs
`ESPnet3LightningTrainer` directly. If a task family needs to normalize
task-specific config before Lightning sees it, subclass
`ESPnet3LightningTrainer`, strip/normalize the extra keys, and delegate the
rest to `super().__init__(...)`:

```python
class MyTaskTrainer(ESPnet3LightningTrainer):
    def __init__(self, model=None, exp_dir=None, config=None, best_model_criterion=None):
        trainer_config = copy.deepcopy(config)
        if isinstance(trainer_config, DictConfig) and hasattr(trainer_config, "my_task_only_key"):
            delattr(trainer_config, "my_task_only_key")

        super().__init__(
            model=model,
            exp_dir=exp_dir,
            config=trainer_config,
            best_model_criterion=best_model_criterion,
        )
```

Keep the base trainer behavior, normalize only task-specific config, and call
`super().__init__(...)` last.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Training configuration"
    desc="See where trainer options are set in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Callbacks"
    desc="See the callback layer attached to the trainer."
    icon="tabler:plug"
    href="./callbacks.html"
  />
  <DocCard
    title="Multiple optimizers"
    desc="See the trainer-side runtime behavior for multi-optimizer training."
    icon="tabler:git-merge"
    href="./multiple_optimizers_schedulers.html"
  />
</DocCards>
