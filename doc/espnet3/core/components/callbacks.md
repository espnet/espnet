---
title: ESPnet3 Callbacks
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Callbacks

ESPnet3's default training callbacks are implemented in:

- `espnet3.components.callbacks.default_callbacks`

The default stack is created by:

- `get_default_callbacks(...)`

## Default callback stack

The current default list is:

- last-checkpoint `ModelCheckpoint`
- best-k `ModelCheckpoint`
- `AverageCheckpointsCallback`
- `LearningRateMonitor`
- `MetricsLogger`
- `TQDMProgressBar`

These are added automatically by `ESPnet3LightningTrainer`.

## MetricsLogger

`MetricsLogger` is the main human-readable training log callback in current
ESPnet3.

Implementation:

- `espnet3.components.callbacks.default_callbacks.MetricsLogger`

**Why it exists**

Instead of scattering summary logging across multiple callbacks or stage code,
ESPnet3 centralizes compact train/validation summaries in one callback.

**What it logs**

`MetricsLogger` handles three reporting points:

- interval train-batch summaries
- end-of-epoch train summaries
- end-of-epoch validation summaries

It also tracks timing-style keys such as:

- `iter_time`
- `forward_time`
- `backward_time`
- `optim_step_time`
- `train_time`
- `valid_time`

and includes optimizer learning rates such as `optim0_lr0`.

**Metric key normalization**

The callback normalizes keys before printing:

- training summaries drop the `train/` prefix
- validation summaries drop the `valid/` prefix
- validation sanity-check runs are ignored

This is why logs stay compact even though the underlying metric names are stored
as `train/...` and `valid/...`.

**Example log lines**

Typical output looks like:

```text
1epoch:train:1-500batch: loss=12.34 iter_time=0.02 forward_time=0.01 backward_time=0.01 optim_step_time=0.00 train_time=0.03 optim0_lr0=0.0020
epoch_summary:1epoch:train: loss=8.91 iter_time=0.02 forward_time=0.01 backward_time=0.01 optim_step_time=0.00 train_time=0.03 optim0_lr0=0.0020
epoch_summary:1epoch:valid: loss=7.85 cer=18.4 valid_time=12.6
```

## AverageCheckpointsCallback

This callback runs on `on_validation_end` (every validation round, on the
global-zero process only). For each `best_model_criterion` entry it:

- loads the state dict from every checkpoint currently in that criterion's
  top-K set,
- averages every matching parameter (integer-type buffers, e.g. BatchNorm's
  `num_batches_tracked`, are accumulated but not averaged),
- writes the result to `<monitor>.ave_<K>best.pth` in `exp_dir`, overwriting
  the previous file.

Because the top-K set is recomputed at every validation end, the file always
reflects the current best-K checkpoints for that criterion. The final training
epoch's checkpoint is included only if its monitored metric is good enough to
be one of the top-K at that point — it is not unconditionally included.

Checkpoint keys are matched against the model's own `state_dict()` (via
`ESPnetLightningModule.state_dict()`, which returns `self.model.state_dict()`
directly, with no `model.` prefix). If a saved checkpoint's keys already match,
they are averaged as-is; otherwise a leading `model.` prefix is stripped from
each checkpoint key before matching again.

## EMACallback (opt-in)

`EMACallback` (`espnet3.components.callbacks.ema.EMACallback`) maintains an
exponential moving average of the model weights. Unlike the callbacks above,
it is **not** part of the default stack — add it explicitly:

```yaml
trainer:
  callbacks:
    - _target_: espnet3.components.callbacks.ema.EMACallback
      decay: 0.9999
```

Behavior:

- updates the EMA copy once per true optimizer step (not per accumulation
  micro-step), tracked via `trainer.global_step`;
- swaps the EMA weights into the online model for `validate`/`test`, and
  restores the online weights afterward;
- saves the EMA state under `ema_model_state_dict` in checkpoints.

**Current limitation:** the EMA holder is only created when `stage == "fit"`
(i.e. inside `trainer.fit(...)`). Calling `trainer.validate(...)` or
`trainer.test(...)` on its own — even when resuming from a checkpoint that
contains `ema_model_state_dict` — does not recreate the EMA holder, so EMA
weights are not swapped in and the saved EMA state is silently ignored. Use
EMA through `fit(...)` if you rely on its validation-time weight swap.

## Extending callbacks

Custom callbacks can still be appended from config:

```yaml
trainer:
  callbacks:
    - _target_: my_project.callbacks.MyCallback
```

The default stack remains, and custom callbacks are appended after it. There
is no config switch to remove or replace a default callback; the list from
`get_default_callbacks(...)` is always prepended.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Trainer"
    desc="See how callbacks are attached and used by the trainer wrapper."
    icon="tabler:player-play"
    href="./trainer.html"
  />
  <DocCard
    title="Training configuration"
    desc="See where callback settings live in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Train stage"
    desc="Return to the stage-level training overview."
    icon="tabler:route"
    href="../../stages/train.html"
  />
</DocCards>
