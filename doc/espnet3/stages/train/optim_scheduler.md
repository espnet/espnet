---
title: ESPnet3 Train Optimizer and Scheduler
---

# ESPnet3 Train Optimizer and Scheduler

This page explains how `optim`/`scheduler` config blocks map to actual
optimizers and schedulers instantiated in the training code.

ESPnet3 reads these configs in `espnet3/components/modeling/lightning_module.py`
(`ESPnetLightningModule.configure_optimizers`)
and instantiates them via Hydra.

You can use any optimizer or scheduler supported by PyTorch. ESPnet2 also ships
custom schedulers; you can reference them the same way with `_target_`.

## ESPnet2 scheduler options

| Class | Description |
| --- | --- |
| [`WarmupLR`](https://espnet.github.io/espnet/guide/espnet2/schedulers/WarmupLR.html) | Linear warmup with constant LR after warmup. |
| [`NoamLR`](https://espnet.github.io/espnet/guide/espnet2/schedulers/NoamLR.html) | Transformer-style warmup + inverse square root decay. |
| [`WarmupStepLR`](https://espnet.github.io/espnet/guide/espnet2/schedulers/WarmupStepLR.html) | Warmup then step-based decay. |
| [`ExponentialDecayWarmup`](https://espnet.github.io/espnet/guide/espnet2/schedulers/ExponentialDecayWarmup.html) | Warmup then exponential decay. |
| [`CosineAnnealingWarmupRestarts`](https://espnet.github.io/espnet/guide/espnet2/schedulers/CosineAnnealingWarmupRestarts.html) | Warmup with cosine annealing and restarts. |
| [`PiecewiseLinearWarmupLR`](https://espnet.github.io/espnet/guide/espnet2/schedulers/PiecewiseLinearWarmupLR.html) | Warmup followed by piecewise linear schedule. |
| [`TristageLR`](https://espnet.github.io/espnet/guide/espnet2/schedulers/TristageLR.html) | Three-stage schedule (warmup, hold, decay). |
| [`WarmupReduceLROnPlateau`](https://espnet.github.io/espnet/guide/espnet2/schedulers/WarmupReduceLROnPlateau.html) | Warmup with ReduceLROnPlateau logic. |

## Single optimizer + scheduler

Use this when the entire model is trained with one optimizer.

```yaml
optim:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.000001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 15000
```

What happens:

- `optim` is instantiated with all trainable parameters.
- `scheduler` is instantiated with the optimizer.

## Multiple optimizers + schedulers

Use this when different model parts use different optimizers/schedulers.
Each entry in `optims` must define:

- `optim`: the optimizer config
- `params`: a substring that matches parameter names (e.g., `encoder`)

```yaml
optims:
  - optim:
      _target_: torch.optim.Adam
      lr: 0.001
    params: encoder
  - optim:
      _target_: torch.optim.SGD
      lr: 0.01
    params: decoder

schedulers:
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 10
  - scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      patience: 2
```

What happens:

- Each `optim` block is instantiated with parameters whose names contain
  the given `params` substring.
- ESPnet3 checks that every trainable parameter is assigned exactly once.
- Schedulers are matched by list index (`schedulers[0]` for `optims[0]`, etc.)
  and wrapped for Lightning.

### Notes

- Do not mix `optim` with `optims`, or `scheduler` with `schedulers`.
- If `params` does not match any parameters, configuration fails.
