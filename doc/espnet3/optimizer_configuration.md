## ESPnet3: Optimizer and Scheduler Configuration

This guide explains how to configure **optimizers and schedulers** in espnet3.

By default, PyTorch Lightning does not support multiple optimizers and schedulers directly.
To address this, ESPnet3 adopts the code originally developed at [this thread](https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073) to enable this functionality.

---

### 1. Single Optimizer Configuration

To configure a single optimizer and scheduler, your config might look like this:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 10
  gamma: 0.1
```

---

### 2. Multiple Optimizers and Schedulers

To define **multiple** optimizers or schedulers, change the config keys from singular to plural (`optimizers`, `schedulers`). For example:

```yaml
optimizers:
  - _target_: torch.optim.Adam
    lr: 0.0005
    params: ["encoder"]

  - _target_: torch.optim.Adam
    lr: 0.001
    params: ["decoder"]

schedulers:
  - _target_: torch.optim.lr_scheduler.StepLR
    step_size: 5
    gamma: 0.5
    optimizer_index: 0

  - _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
    gamma: 0.1
    optimizer_index: 1
```

Each optimizer can be assigned to a subset of model parameters using the `params` key (e.g., "encoder" or "decoder").
Be sure to have the `params` configuration to each optimziers, otherwise we cannot detect which parameters should we assign to which optimizers.

Schedulers use optimizer index to bind to the corresponding optimizer in the optimizers list;
when multiple optimizers are configured (e.g., three), the number of schedulers must match, and each scheduler is associated with its optimizer by their respective positions in the lists - i.e., the first scheduler corresponds to the first optimizer, the second to the second, and so on.
