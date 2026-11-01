---
title: Multiple Optimizers And OptimizationStep In ESPnet3
author:
  name: "Masao Someki"
date: 2026-04-15
---

# Multiple Optimizers And OptimizationStep In ESPnet3

ESPnet3's multi-optimizer path exists for models that should not update all
parameters with one shared loss, especially GAN-style training.

Key implementations:

- `espnet3.components.modeling.optimization_spec.OptimizationStep`
- `espnet3.components.modeling.lightning_module.ESPnetLightningModule`

## Why this exists

In GAN training, generator and discriminator often:

- optimize different losses
- update in different orders
- update at different frequencies
- use different clipping rules

ESPnet3 therefore needs a way to:

- name each optimizer update explicitly
- control update order
- skip some optimizers on some batches
- apply per-optimizer runtime policies

## OptimizationStep

`OptimizationStep` is:

```python
OptimizationStep(loss=<tensor>, name="<optimizer-name>")
```

The `name` must match a key under:

- `optimizers.<name>`
- `schedulers.<name>`

## Model return contract

In the multi-optimizer path, the model returns named optimization steps instead
of one shared loss tensor:

```python
return OptimizationStep(loss=g_loss, name="generator"), stats, weight
```

or:

```python
return [
    OptimizationStep(loss=g_loss, name="generator"),
    OptimizationStep(loss=d_loss, name="discriminator"),
], stats, weight
```

## Ordering controls update order

The order of the returned list is the actual update order for that batch.

Example:

```python
return [
    OptimizationStep(loss=d_loss, name="discriminator"),
    OptimizationStep(loss=g_loss, name="generator"),
], stats, weight
```

means:

1. discriminator backward / step
2. generator backward / step

If you reverse the list, you reverse the update order.

## Omitted optimizers are not updated

If the model returns only:

```python
OptimizationStep(loss=g_loss, name="generator")
```

then the discriminator is untouched for that batch.

This is the current omission semantics. Returning no step for an optimizer means
no backward, no optimizer step, and no scheduler step for that optimizer.

## Per-optimizer update controls

Each named optimizer has its own runtime policy:

- `accum_grad_steps`
- `step_every_n_iters`
- `gradient_clip_val`
- `gradient_clip_algorithm`

The training loop evaluates these per optimizer.

Example:

```yaml
optimizers:
  generator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: generator
    accum_grad_steps: 2
    step_every_n_iters: 1
    gradient_clip_val: 1.0
    gradient_clip_algorithm: norm

  discriminator:
    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0002
    params: discriminator
    accum_grad_steps: 1
    step_every_n_iters: 2
```

That means:

- generator steps every 2 accumulated backward passes
- discriminator steps every other training iteration

## Scheduler behavior

Named schedulers are configured separately:

```yaml
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

Rules:

- step schedulers run after a successful optimizer update
- epoch schedulers run at epoch end
- monitored schedulers use `monitor`

**Prefer `valid/*` monitors.** Training stats are logged once per step and are
not aggregated over the epoch, so a `monitor` such as `train/generator/loss`
reflects only the last training batch when read at `on_train_epoch_end`, not
an epoch average. Use `valid/<name>/loss` (or another `valid/*` key) for
monitored epoch schedulers.

## Parameter selection

Each `optimizers.<name>.params` selector is matched against trainable parameter
names on dot boundaries.

Examples:

- `encoder` matches `model.encoder.layer.weight`
- `encoder.layer1` matches `model.encoder.layer1.weight`
- `encoder` does not match `model.decoder_encoder.weight`

ESPnet3 validates:

- every optimizer matches at least one trainable parameter
- no parameter is assigned twice
- no trainable parameter is left uncovered

## GAN-style example

No recipe currently ships a GAN model out of the box. To write one, follow the
`OptimizationStep` contract shown above: return one `OptimizationStep` per
loss (or a list, when more than one optimizer should update in the same
batch), name each step to match the corresponding `optimizers.<name>` /
`schedulers.<name>` key, and order the list to control update order (e.g.
discriminator first, generator second).

## Practical rules

- use named mappings, not lists
- use `torch.optim.*`
- keep trainer-level gradient clipping disabled in multi-optimizer mode
- return `OptimizationStep` or `list[OptimizationStep]`
- let the list order describe update order

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Optimizer configuration"
    desc="See the broader config surface for optimizer and scheduler setup."
    icon="tabler:adjustments"
    href="./optimizer_configuration.html"
  />
  <DocCard
    title="Trainer"
    desc="See where multiple-optimizer runtime rules are applied."
    icon="tabler:player-play"
    href="./trainer.html"
  />
  <DocCard
    title="Training configuration"
    desc="See where the optimizer mappings are defined in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
</DocCards>
