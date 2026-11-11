---
title: ESPnet3 Optimizer And Scheduler Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Optimizer And Scheduler Configuration

ESPnet3 wraps PyTorch Lightning so that optimizers and schedulers can be
defined from YAML.

Current implementation:

- `espnet3.components.modeling.lightning_module.ESPnetLightningModule.configure_optimizers`
- `espnet3.components.modeling.optimization_spec`

Two modes are supported:

1. single optimizer path: `optimizer` + `scheduler`
2. named multi-optimizer path: `optimizers` + `schedulers`

## What lives in configure_optimizers vs YAML

| Layer | You control via YAML | ESPnet3 ensures |
| --- | --- | --- |
| `optimizer` / `optimizers` | optimizer classes and hyperparameters | correct parameter grouping and uniqueness |
| `scheduler` / `schedulers` | scheduler classes and decay settings | matching schedulers to optimizers |
| model parameter names | which parameters each optimizer sees | every trainable parameter is assigned exactly once |

## 1. Single optimizer path

Use `optimizer` and `scheduler` when the whole model shares one optimizer.

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-2

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100000

scheduler_interval: step
scheduler_monitor:
```

Important points:

- use `torch.optim.*`, not `torch.optimizer.*`
- `optimizer` is instantiated with all trainable parameters
- `scheduler` is instantiated with that optimizer
- `scheduler_interval` must be `step` or `epoch`
- `scheduler_monitor` is only needed for monitored epoch schedulers such as
  `ReduceLROnPlateau`

**`scheduler` is required whenever `optimizer` is set.** There is no
constant-LR fallback for omitting it (or setting `scheduler: null`) — the
single-optimizer branch of `configure_optimizers()` only activates when both
`optimizer` and `scheduler` are set, so leaving `scheduler` empty falls through
to a `ValueError`. For a constant learning rate, configure an explicit no-op
scheduler instead, e.g.:

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.ConstantLR
  factor: 1.0
  total_iters: 1
```

**Use a `valid/*` monitor for epoch-based schedulers.** Training stats are
logged once per step (not aggregated per epoch), while validation stats are
aggregated over the whole epoch. A `scheduler_monitor` such as `train/loss`
would therefore reflect only the last training batch, not an epoch average —
prefer `valid/loss` (or another `valid/*` key) for monitored epoch schedulers.

Example with a monitored scheduler:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  patience: 2
  factor: 0.5

scheduler_interval: epoch
scheduler_monitor: valid/loss
```

## 2. Named multi-optimizer path

Use `optimizers` and `schedulers` when different parameter groups need
independent updates.

This is the normal path for GAN training and other cases where one shared loss
and one shared optimizer are not enough.

Minimal shape:

```yaml
optimizers:
  generator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: generator
    accum_grad_steps: 1
    step_every_n_iters: 1
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm

  discriminator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: discriminator
    accum_grad_steps: 1
    step_every_n_iters: 1

schedulers:
  generator:
    scheduler:
      _target_: torch.optim.lr_scheduler.LinearLR
      start_factor: 1.0
      end_factor: 0.5
      total_iters: 1000
    interval: step

  discriminator:
    scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
      factor: 0.5
    interval: epoch
    monitor: valid/discriminator/loss
```

General rules:

- names under `optimizers` and `schedulers` must match exactly
- every optimizer entry must include `params` and `optimizer`
- top-level `scheduler_interval` and `scheduler_monitor` are not used here
- all detailed per-optimizer routing and runtime rules are documented in
  [Multiple optimizers and `OptimizationStep`](./multiple_optimizers_schedulers.html)

## What not to mix

Do not mix:

- `optimizer` with `optimizers`
- `scheduler` with `schedulers`

ESPnet3 rejects mixed configuration.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Multiple optimizers"
    desc="See the multi-optimizer runtime contract and `OptimizationStep`."
    icon="tabler:git-merge"
    href="./multiple_optimizers_schedulers.html"
  />
  <DocCard
    title="Trainer"
    desc="See where optimizer configuration is consumed by the training wrapper."
    icon="tabler:player-play"
    href="./trainer.html"
  />
  <DocCard
    title="Training configuration"
    desc="See where optimizer and scheduler settings live in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
</DocCards>
